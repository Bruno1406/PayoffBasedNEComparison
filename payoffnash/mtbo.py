from collections.abc import Callable
import csv
import gc
import math
import os
import re
from typing import Any, Optional
import numpy as np
from scipy.stats.qmc import LatinHypercube
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


try:
    from .common import Game, NEEstimator, EstimateHistory, MAX_STEPS
    from .game_ensemble import QuadraticGame
except (ImportError, ValueError):
    try:
        from payoffnash.common import Game, NEEstimator, EstimateHistory, MAX_STEPS
        from payoffnash.game_ensemble import QuadraticGame
    except ImportError:
        from common import Game, NEEstimator, EstimateHistory, MAX_STEPS
        from game_ensemble import QuadraticGame




type MTBOCostFunction = Callable[[QuadraticGame, EstimateHistory], float]
_INDEXED_PARAM_PATTERN = re.compile(r"^([a-zA-Z_]\w*)\[([\d,\s]+)\]$")


class MetaParamOptimization:
    def __init__(
        self,
        game_ensemble: list[QuadraticGame],
        search_space: dict[str, tuple[float, float]],
        algorithm_class: type[NEEstimator],
        cost_function: MTBOCostFunction,
        output_path: Optional[str] = None,
        num_steps: int = MAX_STEPS,
        max_rel_error: float = 0.0,
        time_limit: float = np.inf,
        n_init: Optional[int] = None,
        max_bo_steps: Optional[int] = None,
        min_bo_steps: Optional[int] = None,
        eps_ei: Optional[float] = 1e-4,
        ei_window: int = 3,
        seed: Optional[int] = None,
        fixed_params: Optional[dict[str, Any]] = None,
        initial_actions: Optional[np.ndarray] = None,
    ):
        if not game_ensemble:
            raise ValueError("game_ensemble cannot be empty.")
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if (max_rel_error == 0.0) and num_steps == MAX_STEPS and time_limit == np.inf:
            raise ValueError("At least one stopping criterion must be specified: max_rel_error, num_steps, or time_limit.")

        self.game_ensemble = game_ensemble
        self.search_space = search_space
        self.algorithm_class = algorithm_class
        self.output_path = output_path
        self.num_steps = num_steps
        self.max_rel_error = max_rel_error
        self.time_limit = time_limit

        num_tasks = len(game_ensemble)
        self.n_tasks = num_tasks
        self.param_names = list(search_space.keys())
        self.dim = len(self.param_names)

        if n_init is None:
            self.n_init = 10 if num_tasks == 1 else num_tasks
        else:
            if n_init < 1:
                raise ValueError("n_init must be >= 1.")
            self.n_init = n_init

        if max_bo_steps is None:
            if num_tasks == 1:
                self.max_bo_steps = max(30, 25 * self.dim)
            else:
                self.max_bo_steps = min(num_tasks * max(5, 2 * self.dim), 1500)
        else:
            if max_bo_steps < 0:
                raise ValueError("max_bo_steps must be >= 0.")
            self.max_bo_steps = max_bo_steps

        if min_bo_steps is None:
            self.min_bo_steps = 10 if num_tasks == 1 else max(20, num_tasks // 2)
        else:
            if min_bo_steps < 0:
                raise ValueError("min_bo_steps must be >= 0.")
            self.min_bo_steps = min_bo_steps

        if eps_ei is not None and eps_ei < 0.0:
            raise ValueError("eps_ei must be >= 0.0.")
        self.eps_ei = eps_ei

        if ei_window < 1:
            raise ValueError("ei_window must be >= 1.")
        self.ei_window = ei_window

        self.gc_interval = 100 if num_tasks == 1 else max(10, min(50, 2500 // num_tasks))

        self.n_bo_steps_evaluated = 0
        self.stopped_early = False
        self.stop_reason = ""
        self.task_anchors: Optional[np.ndarray] = None
        self.best_observed_avg: float = float("inf")
        self.seed = seed
        self.fixed_params = fixed_params or {}
        self.initial_actions = initial_actions
        self.cost_function = cost_function

        lower = [search_space[k][0] for k in self.param_names]
        upper = [search_space[k][1] for k in self.param_names]
        self.lower_np = np.array(lower, dtype=float)
        self.upper_np = np.array(upper, dtype=float)

    def _unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return self.lower_np + (self.upper_np - self.lower_np) * x_norm

    @staticmethod
    def _compute_expected_improvement(delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        safe_sigma = torch.clamp(sigma, min=1e-12)
        z = delta / safe_sigma
        norm_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        norm_pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
        return delta * norm_cdf + safe_sigma * norm_pdf

    def _build_estimator_kwargs(self, theta: np.ndarray, game: QuadraticGame) -> dict[str, Any]:
        kwargs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in self.fixed_params.items()}
        param_specs = self.algorithm_class.get_param_specs(game)

        for i, name in enumerate(self.param_names):
            val = float(theta[i])
            match = _INDEXED_PARAM_PATTERN.match(name)
            if match:
                base_name, idx_str = match.groups()
                indices = tuple(int(x.strip()) for x in idx_str.split(","))
                if base_name not in kwargs:
                    shape = param_specs.get(base_name, (game.N, len(game.action_sets[0])))
                    kwargs[base_name] = np.zeros(shape, dtype=float)
                elif not isinstance(kwargs[base_name], np.ndarray):
                    shape = param_specs.get(base_name, (game.N, len(game.action_sets[0])))
                    kwargs[base_name] = np.full(shape, kwargs[base_name], dtype=float)
                kwargs[base_name][indices] = val
            else:
                expected_shape = param_specs.get(name, (1,))
                if expected_shape == (1,):
                    kwargs[name] = val
                else:
                    kwargs[name] = np.full(expected_shape, val, dtype=float)

        return kwargs

    def _flatten_hyperparams(self, theta: np.ndarray, game: QuadraticGame) -> dict[str, Any]:
        kwargs = self._build_estimator_kwargs(theta, game)
        flat = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i in range(len(v)):
                        flat[f"{k}[{i}]"] = float(v[i])
                elif v.ndim == 2:
                    for r in range(v.shape[0]):
                        for c in range(v.shape[1]):
                            flat[f"{k}[{r},{c}]"] = float(v[r, c])
                else:
                    for idx, val in np.ndenumerate(v):
                        flat[f"{k}{list(idx)}"] = float(val)
            elif isinstance(v, (int, float, np.floating)):
                flat[k] = float(v)
            else:
                flat[k] = v
        return flat

    def evaluate_task(self, theta: np.ndarray, task_idx: int) -> float:
        game = self.game_ensemble[task_idx]
        kwargs = self._build_estimator_kwargs(theta, game)
        estimator = self.algorithm_class(game, **kwargs)
        m = len(game.action_sets[0])
        if self.initial_actions is not None:
            init_actions = np.asarray(self.initial_actions, dtype=float).reshape(game.N, m)
        else:
            init_actions = np.array([[0.5 * (iv[0] + iv[1]) for iv in s] for s in game.action_sets], dtype=float)
        res = estimator.estimate(init_actions, num_steps=self.num_steps, max_rel_error=self.max_rel_error, time_limit=self.time_limit)
        return float(self.cost_function(game, res))


    def _sample_and_evaluate_candidates(
        self,
        model: MultiTaskGP,
        sobol_engine: torch.quasirandom.SobolEngine,
        num_cands: int,
        num_tasks: int,
        x_incumbent_unit: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        cands_unit = sobol_engine.draw(n=num_cands).to(dtype=torch.double)
        if x_incumbent_unit is not None:
            cands_unit = torch.cat([x_incumbent_unit.unsqueeze(0), cands_unit], dim=0)
        actual_num_cands = cands_unit.shape[0]

        cands_batched = torch.cat([
            cands_unit.unsqueeze(1).expand(actual_num_cands, num_tasks, self.dim),
            torch.arange(num_tasks, dtype=torch.double).view(1, num_tasks, 1).expand(actual_num_cands, num_tasks, 1),
        ], dim=-1)

        with torch.no_grad():
            post = model.posterior(cands_batched)
            means = post.mean.squeeze(-1)
            cov_b = post.mvn.covariance_matrix
            variances = torch.diagonal(cov_b, dim1=-2, dim2=-1)

            avg_mean = means.mean(dim=-1)
            var_avg = cov_b.sum(dim=(-1, -2)) / (num_tasks ** 2)
            avg_std = torch.sqrt(torch.clamp(var_avg, min=1e-12))
            curr_min_avg_idx = int(torch.argmin(avg_mean).item())

        del cands_batched, post, cov_b
        return cands_unit, means, variances, avg_mean, avg_std, curr_min_avg_idx

    def _extract_final_scores(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        num_tasks: int,
        anchor_thetas: list[np.ndarray],
        task_anchors: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, list[np.ndarray], list[float]]:
        final_model = MultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            outcome_transform=Standardize(m=1),
        )
        mll_final = ExactMarginalLogLikelihood(final_model.likelihood, final_model)
        fit_gpytorch_mll(mll_final, optimizer_kwargs={"options": {"maxiter": 60}})

        unique_unit_x = torch.unique(train_x[:, :-1], dim=0)
        n_unique = unique_unit_x.shape[0]

        pred_means_list = []
        chunk_size = 50
        with torch.no_grad():
            for start_idx in range(0, n_unique, chunk_size):
                end_idx = min(start_idx + chunk_size, n_unique)
                sub_unique = unique_unit_x[start_idx:end_idx]
                n_sub = sub_unique.shape[0]
                sub_eval = torch.cat([
                    sub_unique.repeat_interleave(num_tasks, dim=0),
                    torch.arange(num_tasks, dtype=torch.double).repeat(n_sub).unsqueeze(-1)
                ], dim=-1)
                sub_post = final_model.posterior(sub_eval)
                pred_means_list.append(sub_post.mean.view(n_sub, num_tasks))
            pred_means = torch.cat(pred_means_list, dim=0)

        task_scores = pred_means.clone()
        for idx in range(n_unique):
            u_pt = unique_unit_x[idx]
            match_mask = (train_x[:, :-1] == u_pt).all(dim=-1)
            for j in range(num_tasks):
                task_match = match_mask & (train_x[:, -1] == j)
                if task_match.any():
                    task_scores[idx, j] = train_y[task_match].min()
        avg_scores = task_scores.mean(dim=-1)
        unique_x_np = self._unnormalize(unique_unit_x.cpu().numpy())

        # One-Shot Post-BO Verification Safeguard
        if task_anchors is None:
            task_anchors = self.task_anchors if self.task_anchors is not None else np.full(num_tasks, float("inf"))

        best_thetas: list[np.ndarray] = []
        best_vals: list[float] = []

        for j in range(num_tasks):
            best_idx_j = int(torch.argmin(task_scores[:, j]).item())
            cand_theta_j = unique_x_np[best_idx_j]
            anchor_val_j = float(task_anchors[j])
            anchor_theta_j = anchor_thetas[j]

            u_cand = unique_unit_x[best_idx_j]
            already_eval = ((train_x[:, :-1] == u_cand).all(dim=-1) & (train_x[:, -1] == j))
            if already_eval.any():
                verified_val = float(train_y[already_eval].min().item())
                best_thetas.append(cand_theta_j)
                best_vals.append(verified_val)
            else:
                verified_val = self.evaluate_task(cand_theta_j, j)
                if verified_val <= anchor_val_j:
                    best_thetas.append(cand_theta_j)
                    best_vals.append(verified_val)
                else:
                    best_thetas.append(anchor_theta_j)
                    best_vals.append(anchor_val_j)

        del final_model, mll_final, pred_means_list, pred_means
        gc.collect()

        return unique_x_np, task_scores, avg_scores, best_thetas, best_vals

    def run_optimization_loop(self) -> list[dict[str, Any]]:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        num_tasks = len(self.game_ensemble)
        num_init = max(self.n_init, num_tasks)

        sampler = LatinHypercube(d=self.dim, seed=self.seed)
        samples = sampler.random(n=num_init)

        train_x_list = []
        train_y_list = []

        for i in range(num_init):
            task_idx = i % num_tasks
            theta_orig = self._unnormalize(samples[i])
            y_val = self.evaluate_task(theta_orig, task_idx)

            x_with_task = np.append(samples[i], float(task_idx))
            train_x_list.append(x_with_task)
            train_y_list.append([y_val])

        train_x = torch.tensor(np.array(train_x_list), dtype=torch.double)
        train_y = torch.tensor(np.array(train_y_list), dtype=torch.double)

        num_cands = min(256 * max(2, self.dim), 4096)
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=self.dim,
            scramble=True,
            seed=self.seed + 1000 if self.seed is not None else None,
        )

        task_anchors = torch.full((num_tasks,), float("inf"), dtype=torch.double)
        anchor_thetas: list[Optional[np.ndarray]] = [None] * num_tasks
        for t in range(num_tasks):
            mask = (train_x[:, -1] == t)
            if mask.any():
                min_idx_within = int(torch.argmin(train_y[mask]).item())
                task_anchors[t] = train_y[mask][min_idx_within]
                task_indices = torch.where(mask)[0]
                best_init_row = int(task_indices[min_idx_within].item())
                anchor_thetas[t] = self._unnormalize(train_x[best_init_row, :-1].cpu().numpy())

        x_incumbent_unit = None
        self.stopped_early = False
        self.stop_reason = "max_bo_steps_reached"
        ei_consecutive = 0

        # Persistent surrogate model instantiated once
        model = MultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 30}})

        for step in range(self.max_bo_steps):
            self.n_bo_steps_evaluated = step + 1

            if step > 0:
                if hasattr(model, "outcome_transform") and model.outcome_transform is not None:
                    model.outcome_transform.train()
                    transformed_y, _ = model.outcome_transform(train_y, X=train_x)
                    model.outcome_transform.eval()
                    targets = transformed_y.squeeze(-1)
                else:
                    targets = train_y.squeeze(-1)

                model.set_train_data(train_x, targets, strict=False)
                model.train()
                model.likelihood.train()
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 10}})

            model.eval()
            model.likelihood.eval()

            if num_tasks == 1:
                best_observed = float(train_y.min().item())
                min_y_idx = int(torch.argmin(train_y).item())
                x_incumbent_unit = train_x[min_y_idx, :-1]
            else:
                unique_eval_unit = torch.unique(train_x[:, :-1], dim=0)
                n_eval = unique_eval_unit.shape[0]
                eval_batched = torch.cat([
                    unique_eval_unit.unsqueeze(1).expand(n_eval, num_tasks, self.dim),
                    torch.arange(num_tasks, dtype=torch.double).view(1, num_tasks, 1).expand(n_eval, num_tasks, 1),
                ], dim=-1)
                with torch.no_grad():
                    post_eval = model.posterior(eval_batched)
                    eval_means = post_eval.mean.squeeze(-1)
                    eval_avg_means = eval_means.mean(dim=-1)
                    best_eval_idx = int(torch.argmin(eval_avg_means).item())
                    best_observed = float(eval_avg_means[best_eval_idx].item())
                    x_incumbent_unit = unique_eval_unit[best_eval_idx]
                del eval_batched, post_eval, eval_means, eval_avg_means

            cands_unit, means, variances, avg_mean, avg_std, curr_min_avg_idx = self._sample_and_evaluate_candidates(
                model, sobol_engine, num_cands, num_tasks, x_incumbent_unit
            )

            delta = best_observed - avg_mean
            ei_avg = self._compute_expected_improvement(delta, avg_std)

            diffs = torch.cdist(cands_unit, train_x[:, :-1], p=float("inf"))
            close_mask = (diffs < 1e-5).any(dim=1)
            ei_avg[close_mask] = -float("inf")

            best_cand_idx = int(torch.argmax(ei_avg).item())
            max_ei = float(ei_avg[best_cand_idx].item())
            if max_ei == -float("inf"):
                max_ei = 0.0

            # Mathematical EI convergence check with confirmation window
            if self.eps_ei is not None and (step + 1) >= self.min_bo_steps and max_ei < self.eps_ei:
                ei_consecutive += 1
                if ei_consecutive >= self.ei_window:
                    self.stopped_early = True
                    self.stop_reason = "ei_convergence"
                    print(f"  [BO Early Stop at Step {step + 1}] Mathematical EI convergence: max_ei={max_ei:.6f} < {self.eps_ei} for {self.ei_window} consecutive steps.", flush=True)
                    del cands_unit, means, variances, delta, ei_avg
                    break
            else:
                ei_consecutive = 0

            x_star_unit = cands_unit[best_cand_idx]
            star_means = means[best_cand_idx]
            star_stds = torch.sqrt(torch.clamp(variances[best_cand_idx], min=1e-12))

            delta_t = task_anchors - star_means
            ei_t = self._compute_expected_improvement(delta_t, star_stds)

            if ei_t.max() > 1e-6:
                t_star = int(torch.argmax(ei_t).item())
            else:
                t_star = int(torch.argmax(star_stds).item())

            theta_star = self._unnormalize(x_star_unit.cpu().numpy())
            y_star = self.evaluate_task(theta_star, t_star)

            new_x = torch.cat([x_star_unit, torch.tensor([float(t_star)], dtype=torch.double)]).unsqueeze(0)
            new_y = torch.tensor([[y_star]], dtype=torch.double)

            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)

            if y_star < task_anchors[t_star]:
                task_anchors[t_star] = y_star
                anchor_thetas[t_star] = theta_star

            avg_best_obs = float(task_anchors.mean().item())

            if (step + 1) % 25 == 0 or step == 0 or step == self.max_bo_steps - 1:
                ei_info = f" | max_ei: {max_ei:.6f}" if self.eps_ei is not None else ""
                if num_tasks == 1:
                    print(f"  [BO Step {step + 1}/{self.max_bo_steps}] Task 0: loss={y_star:.5f}, best_loss={task_anchors[0].item():.5f}{ei_info}", flush=True)
                else:
                    print(f"  [BO Step {step + 1}/{self.max_bo_steps}] Task {t_star}: loss={y_star:.5f}, task_best={task_anchors[t_star].item():.5f}, avg_best={best_observed:.5f}{ei_info}", flush=True)

            del cands_unit, means, variances, delta, ei_avg
            if (step + 1) % self.gc_interval == 0:
                gc.collect()

        del model, mll
        gc.collect()

        self.task_anchors = task_anchors.cpu().numpy()
        self.best_observed_avg = float(np.mean(self.task_anchors))

        self.train_x = train_x
        self.train_y = train_y

        unique_x_np, task_scores, avg_scores, best_thetas, best_vals = self._extract_final_scores(
            train_x, train_y, num_tasks, anchor_thetas, self.task_anchors
        )

        results: list[dict[str, Any]] = []

        self.best_thetas = best_thetas
        for j in range(num_tasks):
            best_theta_j = best_thetas[j]
            best_val_j = best_vals[j]

            opt_dict_j = self._flatten_hyperparams(best_theta_j, self.game_ensemble[j])
            opt_dict_j["game_id"] = f"game_{j}"
            opt_dict_j["is_default"] = False
            opt_dict_j["objective_value"] = best_val_j
            results.append(opt_dict_j)

        best_idx_def = int(torch.argmin(avg_scores).item())
        best_theta_def = unique_x_np[best_idx_def]
        best_val_def = float(avg_scores[best_idx_def].item())

        opt_dict_def = self._flatten_hyperparams(best_theta_def, self.game_ensemble[0])
        opt_dict_def["game_id"] = "default"
        opt_dict_def["is_default"] = True
        opt_dict_def["objective_value"] = best_val_def
        results.append(opt_dict_def)
        self.best_theta_def = best_theta_def

        if self.output_path is not None:
            self._save_csv(results)

        return results

    def build_estimator(self, theta: np.ndarray, game: QuadraticGame) -> NEEstimator:
        kwargs = self._build_estimator_kwargs(theta, game)
        return self.algorithm_class(game, **kwargs)

    def _save_csv(self, results: list[dict[str, Any]]) -> None:
        if not results:
            return

        param_keys = [k for k in results[0].keys() if k not in ("game_id", "is_default", "objective_value")]
        fieldnames = ["game_id", "is_default"] + param_keys + ["objective_value"]

        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
