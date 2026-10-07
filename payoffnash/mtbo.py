import csv
import os
from typing import Any, Optional
import numpy as np
from scipy.stats.qmc import LatinHypercube
import torch
import math
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


try:
    from .common import Game, NEEstimator
    from .game_ensemble import QuadraticGame
except (ImportError, ValueError):
    try:
        from payoffnash.common import Game, NEEstimator
        from payoffnash.game_ensemble import QuadraticGame
    except ImportError:
        from common import Game, NEEstimator
        from game_ensemble import QuadraticGame


class MetaParamOptimization:
    def __init__(
        self,
        game_ensemble: list[QuadraticGame],
        search_space: dict[str, tuple[float, float]],
        algorithm_class: type[NEEstimator],
        output_path: Optional[str] = None,
        num_iters: int = 1000,
        smoothing_window: int = 100,
        n_init: int = 10,
        n_bo_steps: int = 20,
        seed: Optional[int] = None,
        fixed_params: Optional[dict[str, Any]] = None,
    ):
        if not game_ensemble:
            raise ValueError("game_ensemble cannot be empty.")
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        if n_init < 1:
            raise ValueError("n_init must be >= 1.")
        if n_bo_steps < 0:
            raise ValueError("n_bo_steps must be >= 0.")
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1.")

        self.game_ensemble = game_ensemble
        self.search_space = search_space
        self.algorithm_class = algorithm_class
        self.output_path = output_path
        self.num_iters = num_iters
        self.smoothing_window = smoothing_window
        self.n_init = n_init
        self.n_bo_steps = n_bo_steps
        self.seed = seed
        self.fixed_params = fixed_params or {}

        self.param_names = list(search_space.keys())
        self.dim = len(self.param_names)

        lower = [search_space[k][0] for k in self.param_names]
        upper = [search_space[k][1] for k in self.param_names]
        self.bounds = torch.tensor([lower, upper], dtype=torch.double)
        self.unit_bounds = torch.stack(
            [torch.zeros(self.dim, dtype=torch.double), torch.ones(self.dim, dtype=torch.double)]
        )

    def _unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        lower_np = self.bounds[0].cpu().numpy()
        upper_np = self.bounds[1].cpu().numpy()
        return lower_np + (upper_np - lower_np) * x_norm

    def _build_estimator_kwargs(self, theta: np.ndarray, game: QuadraticGame) -> dict[str, Any]:
        kwargs = dict(self.fixed_params)
        param_specs = self.algorithm_class.get_param_specs(game)

        for i, name in enumerate(self.param_names):
            val = float(theta[i])
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
        init_actions = np.zeros((game.N, m), dtype=float)

        _, action_history, _ = estimator.estimate(init_actions, num_iters=self.num_iters)
        x_star = game.nash_equilibrium.ravel()

        dists = np.linalg.norm(action_history - x_star, axis=-1)
        k_window = min(self.smoothing_window, len(dists))
        return float(np.mean(dists[-k_window:]))

    def evaluate_ensemble(self, theta: np.ndarray) -> np.ndarray:
        return np.array([self.evaluate_task(theta, j) for j in range(len(self.game_ensemble))], dtype=float)

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

        num_cands = 64
        cand_sampler = LatinHypercube(d=self.dim, seed=self.seed + 999 if self.seed is not None else None)
        cands_unit = torch.tensor(cand_sampler.random(n=num_cands), dtype=torch.double)
        cands_rep = cands_unit.repeat_interleave(num_tasks, dim=0)
        task_col = torch.arange(num_tasks, dtype=torch.double).repeat(num_cands).unsqueeze(-1)
        cands_all_tasks = torch.cat([cands_rep, task_col], dim=-1)

        for step in range(self.n_bo_steps):
            model = MultiTaskGP(
                train_x,
                train_y,
                task_feature=-1,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 30}})

            with torch.no_grad():
                post = model.posterior(cands_all_tasks)
                means = post.mean.view(num_cands, num_tasks)
                variances = post.variance.view(num_cands, num_tasks)

                avg_mean = means.mean(dim=-1)
                avg_std = torch.sqrt(torch.clamp(variances.mean(dim=-1), min=1e-12))

                best_avg = avg_mean.min()
                delta = best_avg - avg_mean
                z = delta / avg_std
                norm_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
                norm_pdf = torch.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
                ei_avg = delta * norm_cdf + avg_std * norm_pdf

                best_cand_idx = int(torch.argmax(ei_avg).item())
                x_star_unit = cands_unit[best_cand_idx]

                cand_task_input = torch.cat([
                    x_star_unit.unsqueeze(0).repeat(num_tasks, 1),
                    torch.arange(num_tasks, dtype=torch.double).unsqueeze(-1)
                ], dim=-1)
                post_star = model.posterior(cand_task_input)
                star_means = post_star.mean.squeeze(-1)
                star_stds = torch.sqrt(torch.clamp(post_star.variance.squeeze(-1), min=1e-12))

                task_best_vals = torch.empty(num_tasks, dtype=torch.double)
                for t in range(num_tasks):
                    mask = (train_x[:, -1] == t)
                    if mask.any():
                        task_best_vals[t] = train_y[mask].min()
                    else:
                        task_best_vals[t] = train_y.min()

                delta_t = task_best_vals - star_means
                z_t = delta_t / star_stds
                norm_cdf_t = 0.5 * (1.0 + torch.erf(z_t / math.sqrt(2.0)))
                norm_pdf_t = torch.exp(-0.5 * z_t ** 2) / math.sqrt(2.0 * math.pi)
                ei_t = delta_t * norm_cdf_t + star_stds * norm_pdf_t

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

            if (step + 1) % 25 == 0 or step == 0 or step == self.n_bo_steps - 1:
                print(f"  [BO Step {step + 1}/{self.n_bo_steps}] Task {t_star}: loss={y_star:.5f}, best predicted avg={float(best_avg):.5f}", flush=True)


        final_model = MultiTaskGP(
            train_x,
            train_y,
            task_feature=-1,
            outcome_transform=Standardize(m=1),
        )
        mll_final = ExactMarginalLogLikelihood(final_model.likelihood, final_model)
        fit_gpytorch_mll(mll_final, optimizer_kwargs={"options": {"maxiter": 30}})

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
            avg_pred_means = pred_means.mean(dim=-1)

        unique_x_np = self._unnormalize(unique_unit_x.cpu().numpy())

        results: list[dict[str, Any]] = []

        self.best_thetas = []
        for j in range(num_tasks):
            best_idx_j = int(torch.argmin(pred_means[:, j]).item())
            best_theta_j = unique_x_np[best_idx_j]
            best_val_j = float(pred_means[best_idx_j, j].item())

            opt_dict_j = self._flatten_hyperparams(best_theta_j, self.game_ensemble[j])
            opt_dict_j["game_id"] = f"game_{j}"
            opt_dict_j["is_default"] = False
            opt_dict_j["objective_value"] = best_val_j
            results.append(opt_dict_j)
            self.best_thetas.append(best_theta_j)

        best_idx_def = int(torch.argmin(avg_pred_means).item())
        best_theta_def = unique_x_np[best_idx_def]
        best_val_def = float(avg_pred_means[best_idx_def].item())

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
