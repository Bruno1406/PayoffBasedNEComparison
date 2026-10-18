import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.jit._script")
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script is not supported in Python 3\.14.*")

from collections.abc import Callable
import concurrent.futures
import csv
import gc
import os
import re
from typing import Any, Optional, Union
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize

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


def _worker_simulate(
    algorithm_class: type[NEEstimator],
    game: QuadraticGame,
    kwargs: dict[str, Any],
    initial_actions: Optional[np.ndarray],
    num_steps: int,
    max_rel_error: float,
    time_limit: float,
    cost_function: Union[MTBOCostFunction, bytes],
    sim_per_param: int = 1,
    base_seed: Optional[int] = None,
) -> float:
    if isinstance(cost_function, bytes):
        import cloudpickle
        cost_function = cloudpickle.loads(cost_function)
    m = len(game.action_sets[0])
    if initial_actions is not None:
        init_actions = np.asarray(initial_actions, dtype=float).reshape(game.N, m)
    else:
        init_actions = np.array([[0.5 * (iv[0] + iv[1]) for iv in s] for s in game.action_sets], dtype=float)

    if sim_per_param <= 1:
        if base_seed is not None:
            np.random.seed(base_seed)
        estimator = algorithm_class(game, **kwargs)
        res = estimator.estimate(init_actions, num_steps=num_steps, max_rel_error=max_rel_error, time_limit=time_limit)
        val = float(cost_function(game, res))
        return val if np.isfinite(val) else 1e6

    losses: list[float] = []
    for k in range(sim_per_param):
        if base_seed is not None:
            np.random.seed(base_seed + k)
        estimator = algorithm_class(game, **kwargs)
        res = estimator.estimate(init_actions, num_steps=num_steps, max_rel_error=max_rel_error, time_limit=time_limit)
        val = float(cost_function(game, res))
        losses.append(val if np.isfinite(val) else 1e6)
    return float(np.mean(losses))


def _worker_simulate_unpack(args: tuple) -> float:
    return _worker_simulate(*args)


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
        max_bo_steps: int = 200,
        min_bo_steps: int = 20,
        eps_ei: Optional[float] = 1e-4,
        patience: int = 6,
        tol_stagnation: float = 1e-4,
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
        fixed_params: Optional[dict[str, Any]] = None,
        scaled_params: Optional[dict[str, str]] = None,
        initial_actions: Optional[np.ndarray] = None,
        sim_per_param: int = 1,
    ):
        if not game_ensemble:
            raise ValueError("game_ensemble cannot be empty.")
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        if (max_rel_error == 0.0) and num_steps == MAX_STEPS and time_limit == np.inf:
            raise ValueError("At least one stopping criterion must be specified: max_rel_error, num_steps, or time_limit.")
        if max_bo_steps < 0:
            raise ValueError("max_bo_steps must be >= 0.")
        if eps_ei is not None and eps_ei < 0.0:
            raise ValueError("eps_ei must be >= 0.0.")
        if patience < 1:
            raise ValueError("patience must be >= 1.")

        self.game_ensemble = game_ensemble
        self.search_space = search_space
        self.algorithm_class = algorithm_class
        self.cost_function = cost_function
        self.output_path = output_path
        self.num_steps = num_steps
        self.max_rel_error = max_rel_error
        self.time_limit = time_limit

        self.n_tasks = len(game_ensemble)
        self.param_names = list(search_space.keys())
        self.dim = len(self.param_names)

        self.n_init = max(10, 2 * self.dim) if n_init is None else max(1, n_init)
        self.max_bo_steps = max_bo_steps
        self.min_bo_steps = max(0, min_bo_steps)
        self.eps_ei = eps_ei
        self.patience = patience
        self.tol_stagnation = tol_stagnation
        self.seed = seed
        self.fixed_params = fixed_params or {}
        self.scaled_params = scaled_params or {}
        self.initial_actions = initial_actions
        self.sim_per_param = max(1, sim_per_param)

        if n_jobs is None or n_jobs < 0:
            self.n_jobs = min(os.cpu_count() or 4, self.n_tasks) if self.n_tasks > 2 else 1
        else:
            self.n_jobs = max(1, n_jobs)

        lower = [search_space[k][0] for k in self.param_names]
        upper = [search_space[k][1] for k in self.param_names]
        self.lower_np = np.array(lower, dtype=float)
        self.upper_np = np.array(upper, dtype=float)

        self.is_log = {k: (search_space[k][0] > 0.0 and search_space[k][1] > 0.0) for k in self.param_names}
        self.log_lower = np.array(
            [np.log10(search_space[k][0]) if self.is_log[k] else search_space[k][0] for k in self.param_names],
            dtype=float,
        )
        self.log_upper = np.array(
            [np.log10(search_space[k][1]) if self.is_log[k] else search_space[k][1] for k in self.param_names],
            dtype=float,
        )

        self.n_bo_steps_evaluated = 0
        self.stopped_early = False
        self.stop_reason = ""
        self.best_thetas: list[np.ndarray] = []
        self.best_theta_def: Optional[np.ndarray] = None
        self.train_x: Optional[torch.Tensor] = None
        self.train_y: Optional[torch.Tensor] = None
        self.last_model: Optional[MultiTaskGP] = None
        self.anomaly_count = 0
        self.refinements_executed = 0
        self.validation_metrics: Optional[dict[str, Any]] = None

    def _unnormalize(self, x_unit: np.ndarray) -> np.ndarray:
        x_clamped = np.clip(np.asarray(x_unit, dtype=float), 0.0, 1.0)
        scaled = self.log_lower + x_clamped * (self.log_upper - self.log_lower)
        orig = np.empty_like(scaled)
        for i, k in enumerate(self.param_names):
            orig[i] = 10.0 ** scaled[i] if self.is_log[k] else scaled[i]
        return np.clip(orig, self.lower_np, self.upper_np)

    def _normalize(self, x_orig: np.ndarray) -> np.ndarray:
        scaled = np.empty_like(x_orig)
        for i, k in enumerate(self.param_names):
            val = float(x_orig[i])
            scaled[i] = np.log10(np.clip(val, 1e-12, None)) if self.is_log[k] else val
        return np.clip((scaled - self.log_lower) / (self.log_upper - self.log_lower), 0.0, 1.0)

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

        if self.scaled_params:
            bounds = np.array([[iv[1] - iv[0] for iv in s] for s in game.action_sets], dtype=float)
            min_w = float(np.min(bounds))
            for param_key, scale_mode in self.scaled_params.items():
                if scale_mode == "min_box_width":
                    if param_key in kwargs:
                        kwargs[param_key] = kwargs[param_key] * min_w
                    else:
                        match = _INDEXED_PARAM_PATTERN.match(param_key)
                        if match:
                            base_name, idx_str = match.groups()
                            indices = tuple(int(x.strip()) for x in idx_str.split(","))
                            if base_name in kwargs and isinstance(kwargs[base_name], np.ndarray):
                                kwargs[base_name][indices] = kwargs[base_name][indices] * min_w
                elif scale_mode in ("player_box_width", "box_width"):
                    match = _INDEXED_PARAM_PATTERN.match(param_key)
                    if match:
                        base_name, idx_str = match.groups()
                        indices = tuple(int(x.strip()) for x in idx_str.split(","))
                        p_idx = indices[0]
                        dim_idx = indices[1] if len(indices) > 1 else 0
                        w_p = float(bounds[p_idx, dim_idx])
                        if base_name in kwargs and isinstance(kwargs[base_name], np.ndarray):
                            kwargs[base_name][indices] = kwargs[base_name][indices] * w_p
                    elif param_key in kwargs and isinstance(kwargs[param_key], np.ndarray):
                        kwargs[param_key] = kwargs[param_key] * bounds.reshape(kwargs[param_key].shape)

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

    @staticmethod
    def _compute_analytic_ei(delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma_clamped = torch.clamp(sigma, min=1e-8)
        z = delta / sigma_clamped
        cdf = 0.5 * (1.0 + torch.erf(z / 1.4142135623730951))
        pdf = torch.exp(-0.5 * z ** 2) / 2.5066282746310002
        return delta * cdf + sigma_clamped * pdf

    @staticmethod
    def _compute_stagnation_and_patience(history: list[float], patience: int, tol: float) -> tuple[int, int]:
        if len(history) <= 1:
            return 0, patience
        current_best = history[-1]
        stagnant_steps = 0
        for prev_val in reversed(history[:-1]):
            if prev_val - current_best < tol:
                stagnant_steps += 1
            else:
                break
        rem_patience = max(0, patience - stagnant_steps)
        return stagnant_steps, rem_patience

    @staticmethod
    def _garbage_collect() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def evaluate_task(self, theta: np.ndarray, task_idx: int, sim_per_param: Optional[int] = None) -> float:
        game = self.game_ensemble[task_idx]
        kwargs = self._build_estimator_kwargs(theta, game)
        base_seed = (self.seed + 1000 * task_idx) if self.seed is not None else None
        effective_sim = self.sim_per_param if sim_per_param is None else max(1, sim_per_param)
        return _worker_simulate(
            self.algorithm_class,
            game,
            kwargs,
            self.initial_actions,
            self.num_steps,
            self.max_rel_error,
            self.time_limit,
            self.cost_function,
            effective_sim,
            base_seed,
        )

    def evaluate_batch(
        self,
        requests: list[tuple[np.ndarray, int]],
        sim_per_param: Optional[int] = None,
    ) -> list[float]:
        if not requests:
            return []
        effective_sim = self.sim_per_param if sim_per_param is None else max(1, sim_per_param)
        if self.n_jobs == 1 or len(requests) <= 1:
            return [self.evaluate_task(th, j, sim_per_param=effective_sim) for th, j in requests]

        cost_payload: Any = self.cost_function
        if getattr(self.cost_function, "__module__", None) == "__main__":
            import cloudpickle
            cost_payload = cloudpickle.dumps(self.cost_function)

        args_list = [
            (
                self.algorithm_class,
                self.game_ensemble[j],
                self._build_estimator_kwargs(th, self.game_ensemble[j]),
                self.initial_actions,
                self.num_steps,
                self.max_rel_error,
                self.time_limit,
                cost_payload,
                effective_sim,
                (self.seed + 1000 * j) if self.seed is not None else None,
            )
            for th, j in requests
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            return list(executor.map(_worker_simulate_unpack, args_list))

    def evaluate_ensemble(self, theta: np.ndarray, sim_per_param: Optional[int] = None) -> np.ndarray:
        requests = [(theta, j) for j in range(self.n_tasks)]
        losses = self.evaluate_batch(requests, sim_per_param=sim_per_param)
        return np.asarray(losses, dtype=float)

    def run_optimization_loop(self, sim_per_param: Optional[int] = None) -> list[dict[str, Any]]:
        if sim_per_param is not None:
            self.sim_per_param = max(1, sim_per_param)

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        num_tasks = self.n_tasks
        sobol = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
        init_unit_samples = sobol.draw(self.n_init).to(dtype=torch.double).numpy()

        train_x_list: list[torch.Tensor] = []
        train_y_list: list[torch.Tensor] = []
        history_by_game: dict[int, list[tuple[np.ndarray, float]]] = {j: [] for j in range(num_tasks)}
        shared_avg_history: list[tuple[np.ndarray, float]] = []

        init_requests: list[tuple[np.ndarray, int]] = []
        for i in range(self.n_init):
            theta_i = self._unnormalize(init_unit_samples[i])
            for j in range(num_tasks):
                init_requests.append((theta_i, j))

        init_losses = self.evaluate_batch(init_requests)

        for i in range(self.n_init):
            u_i = init_unit_samples[i]
            theta_i = self._unnormalize(u_i)
            block_losses = init_losses[i * num_tasks : (i + 1) * num_tasks]
            shared_avg_history.append((theta_i, float(np.mean(block_losses))))
            for j in range(num_tasks):
                loss_ij = float(block_losses[j])
                history_by_game[j].append((theta_i, loss_ij))
                row_x = torch.cat([torch.tensor(u_i, dtype=torch.double), torch.tensor([float(j)], dtype=torch.double)])
                train_x_list.append(row_x.unsqueeze(0))
                train_y_list.append(torch.tensor([[loss_ij]], dtype=torch.double))

        train_x = torch.cat(train_x_list, dim=0)
        train_y = torch.cat(train_y_list, dim=0)

        active_tasks = set(range(num_tasks))
        def_active = True
        prev_state_dict: Optional[dict[str, Any]] = None
        self.stopped_early = False
        self.stop_reason = "max_bo_steps_reached"
        self.n_bo_steps_evaluated = 0

        best_avg_history: list[float] = []
        best_tailored_history: dict[int, list[float]] = {j: [] for j in range(num_tasks)}

        n_cands_pool = min(4096, max(256, 128 * self.dim))

        for step in range(self.max_bo_steps):
            self.n_bo_steps_evaluated = step + 1
            print(f"[BO Step {step + 1}/{self.max_bo_steps}] Active games: {len(active_tasks)}/{num_tasks} | Def active: {def_active}", flush=True)

            model = MultiTaskGP(train_x, train_y, task_feature=-1, outcome_transform=Standardize(m=1))
            if prev_state_dict is not None:
                try:
                    model.load_state_dict(prev_state_dict, strict=False)
                except Exception:
                    pass

            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 25}})
            prev_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.last_model = model

            mc_seed = self.seed + step + 1 if self.seed is not None else None
            sobol_pool = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True, seed=mc_seed)
            cand_pool = sobol_pool.draw(n_cands_pool).to(dtype=torch.double)

            with torch.no_grad():
                post = model.posterior(cand_pool.unsqueeze(1))
                post_mean = post.mean.squeeze(1)
                post_var = post.variance.squeeze(1)
                post_sigma = torch.sqrt(torch.clamp(post_var, min=1e-8))

            if hasattr(model, "prediction_strategy") and model.prediction_strategy is not None:
                model.prediction_strategy = None
            del mll, post

            cands_ind: dict[int, np.ndarray] = {}
            eis_ind: dict[int, float] = {}

            # Individual active tasks acquisition
            for j in sorted(active_tasks):
                best_val_j = min([pt[1] for pt in history_by_game[j]])
                delta_j = best_val_j - post_mean[:, j]
                ei_pool_j = self._compute_analytic_ei(delta_j, post_sigma[:, j])
                best_idx_j = int(torch.argmax(ei_pool_j).item())
                best_ei_j = float(ei_pool_j[best_idx_j].item())
                cands_ind[j] = cand_pool[best_idx_j].cpu().numpy()
                eis_ind[j] = best_ei_j

            # Universal default candidate acquisition
            cand_avg_unit: Optional[np.ndarray] = None
            best_t: Optional[int] = None
            ei_avg_val: float = 0.0

            if def_active:
                best_val_avg = min([pt[1] for pt in shared_avg_history])
                avg_mean = post_mean.mean(dim=-1)
                avg_sigma = torch.sqrt(torch.clamp((post_sigma ** 2).sum(dim=-1), min=1e-8)) / num_tasks
                delta_avg = best_val_avg - avg_mean
                ei_pool_avg = self._compute_analytic_ei(delta_avg, avg_sigma)
                best_idx_avg = int(torch.argmax(ei_pool_avg).item())
                ei_avg_val = float(ei_pool_avg[best_idx_avg].item())
                cand_avg_unit = cand_pool[best_idx_avg].cpu().numpy()

                best_vals_all = torch.tensor([min([pt[1] for pt in history_by_game[t]]) for t in range(num_tasks)], dtype=torch.double)
                deltas_at_cand = best_vals_all - post_mean[best_idx_avg]
                eis_at_cand = self._compute_analytic_ei(deltas_at_cand, post_sigma[best_idx_avg])
                best_t = int(torch.argmax(eis_at_cand).item())

            # Individual task pruning
            tasks_to_prune: set[int] = set()
            for j in sorted(active_tasks):
                best_val_j = min([pt[1] for pt in history_by_game[j]])
                best_tailored_history[j].append(best_val_j)
                pruned_ei = (self.eps_ei is not None and eis_ind[j] < self.eps_ei)
                pruned_stagnation = False
                imp_j = 0.0
                if len(best_tailored_history[j]) >= self.patience:
                    imp_j = best_tailored_history[j][-self.patience] - best_tailored_history[j][-1]
                    pruned_stagnation = (imp_j < self.tol_stagnation)
                if (step + 1 >= self.min_bo_steps) and (pruned_ei or pruned_stagnation):
                    tasks_to_prune.add(j)
                    if pruned_ei and pruned_stagnation:
                        reason = f"Expected Improvement (EI = {eis_ind[j]:.6e} < {self.eps_ei:.6e}) and Incumbent Stagnation (imp over {self.patience} steps = {imp_j:.6e} < {self.tol_stagnation:.6e})"
                    elif pruned_ei:
                        reason = f"Expected Improvement floor (EI = {eis_ind[j]:.6e} < {self.eps_ei:.6e})"
                    else:
                        reason = f"Incumbent Stagnation (imp over {self.patience} steps = {imp_j:.6e} < {self.tol_stagnation:.6e}, EI at prune = {eis_ind[j]:.6e})"
                    print(f"[Task Pruned] Game {j}: {reason}", flush=True)

            for j in tasks_to_prune:
                active_tasks.remove(j)

            # Universal candidate pruning
            if def_active:
                best_val_avg = min([pt[1] for pt in shared_avg_history])
                best_avg_history.append(best_val_avg)
                pruned_def_ei = (self.eps_ei is not None and ei_avg_val < self.eps_ei)
                pruned_def_stagnation = False
                imp_def = 0.0
                if len(best_avg_history) >= self.patience:
                    imp_def = best_avg_history[-self.patience] - best_avg_history[-1]
                    pruned_def_stagnation = (imp_def < self.tol_stagnation)
                if (step + 1 >= self.min_bo_steps) and (pruned_def_ei or pruned_def_stagnation):
                    def_active = False
                    if pruned_def_ei and pruned_def_stagnation:
                        reason_def = f"Expected Improvement (EI = {ei_avg_val:.6e} < {self.eps_ei:.6e}) and Incumbent Stagnation (imp over {self.patience} steps = {imp_def:.6e} < {self.tol_stagnation:.6e})"
                    elif pruned_def_ei:
                        reason_def = f"Expected Improvement floor (EI = {ei_avg_val:.6e} < {self.eps_ei:.6e})"
                    else:
                        reason_def = f"Incumbent Stagnation (imp over {self.patience} steps = {imp_def:.6e} < {self.tol_stagnation:.6e}, EI at prune = {ei_avg_val:.6e})"
                    print(f"[Task Pruned] Universal Default candidate: {reason_def}", flush=True)

            # Check global halting
            if (step + 1 >= self.min_bo_steps) and (not def_active and len(active_tasks) == 0):
                self.stopped_early = True
                self.stop_reason = "all_tasks_pruned"
                print(f"\n[BO Loop Complete] All M+1 tasks pruned at step {step + 1}. Halting optimization loop.\n", flush=True)
                break

            # 5-Iteration Progress Summary
            if (step + 1) % 5 == 0:
                print(f"\n{'='*25} BO Step {step + 1}/{self.max_bo_steps} Progress Summary {'='*25}")
                print(f"{'Task':<22} | {'Status':<8} | {'Best Loss':<14} | {'Current EI':<14} | {'Patience'}")
                print("-" * 75)
                _, rem_pat_def = self._compute_stagnation_and_patience(best_avg_history, self.patience, self.tol_stagnation)
                status_def = "ACTIVE" if def_active else "PRUNED"
                loss_def = min([pt[1] for pt in shared_avg_history])
                ei_def = ei_avg_val if def_active else 0.0
                pat_str_def = f"{rem_pat_def}/{self.patience}" if def_active else f"0/{self.patience}"
                print(f"{'Universal Default':<22} | {status_def:<8} | {loss_def:<14.6e} | {ei_def:<14.6e} | {pat_str_def}")

                for j in range(num_tasks):
                    is_active = (j in active_tasks)
                    status_j = "ACTIVE" if is_active else "PRUNED"
                    loss_j = min([pt[1] for pt in history_by_game[j]])
                    ei_j = eis_ind.get(j, 0.0) if is_active else 0.0
                    _, rem_pat_j = self._compute_stagnation_and_patience(best_tailored_history[j], self.patience, self.tol_stagnation)
                    pat_str_j = f"{rem_pat_j}/{self.patience}" if is_active else f"0/{self.patience}"
                    print(f"{f'Game {j}':<22} | {status_j:<8} | {loss_j:<14.6e} | {ei_j:<14.6e} | {pat_str_j}")
                print("-" * 75)
                print(f"Summary: {len(active_tasks)}/{num_tasks} active games | Universal Default: {status_def}\n", flush=True)

            # Batch simulation dispatch
            batch_requests: list[tuple[np.ndarray, int]] = []
            cand_map: dict[tuple[int, tuple[float, ...]], float] = {}

            if def_active and cand_avg_unit is not None and best_t is not None:
                th_avg = self._unnormalize(cand_avg_unit)
                batch_requests.append((th_avg, best_t))

            for j in sorted(active_tasks):
                th_j = self._unnormalize(cands_ind[j])
                if not (def_active and j == best_t and np.allclose(cands_ind[j], cand_avg_unit)):
                    batch_requests.append((th_j, j))

            losses = self.evaluate_batch(batch_requests)

            new_x_step: list[torch.Tensor] = []
            new_y_step: list[torch.Tensor] = []
            req_idx = 0

            if def_active and cand_avg_unit is not None and best_t is not None:
                loss_t = losses[req_idx]
                history_by_game[best_t].append((th_avg, loss_t))
                row_x = torch.cat([torch.tensor(cand_avg_unit, dtype=torch.double), torch.tensor([float(best_t)], dtype=torch.double)])
                new_x_step.append(row_x.unsqueeze(0))
                new_y_step.append(torch.tensor([[loss_t]], dtype=torch.double))
                cand_map[(best_t, tuple(np.round(cand_avg_unit, 8)))] = loss_t
                req_idx += 1

            for j in sorted(active_tasks):
                cand_u = cands_ind[j]
                th_j = self._unnormalize(cand_u)
                cand_key = (j, tuple(np.round(cand_u, 8)))
                if cand_key in cand_map:
                    loss_j = cand_map[cand_key]
                else:
                    loss_j = losses[req_idx]
                    req_idx += 1
                    row_x = torch.cat([torch.tensor(cand_u, dtype=torch.double), torch.tensor([float(j)], dtype=torch.double)])
                    new_x_step.append(row_x.unsqueeze(0))
                    new_y_step.append(torch.tensor([[loss_j]], dtype=torch.double))
                history_by_game[j].append((th_j, loss_j))

            if new_x_step:
                train_x = torch.cat([train_x] + new_x_step, dim=0)
                train_y = torch.cat([train_y] + new_y_step, dim=0)

            self._garbage_collect()

        self.train_x = train_x
        self.train_y = train_y

        return self._extract_results_with_refinement(history_by_game, train_x, train_y)

    def _extract_results_with_refinement(
        self,
        history_by_game: dict[int, list[tuple[np.ndarray, float]]],
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> list[dict[str, Any]]:
        num_tasks = self.n_tasks
        all_evaluated_thetas = []
        for j in range(num_tasks):
            all_evaluated_thetas.extend([pt[0] for pt in history_by_game[j]])

        unique_thetas_np = np.unique(np.array(all_evaluated_thetas), axis=0)
        unique_thetas_unit = torch.tensor(
            np.array([self._normalize(th) for th in unique_thetas_np]),
            dtype=torch.double,
        )

        n_unique = len(unique_thetas_unit)
        pred_means_list = []
        chunk_size = 64

        with torch.no_grad():
            for start_idx in range(0, n_unique, chunk_size):
                end_idx = min(start_idx + chunk_size, n_unique)
                sub_unit = unique_thetas_unit[start_idx:end_idx]
                n_sub = len(sub_unit)
                sub_eval = torch.cat([
                    sub_unit.repeat_interleave(num_tasks, dim=0),
                    torch.arange(num_tasks, dtype=torch.double).repeat(n_sub).unsqueeze(-1)
                ], dim=-1)
                sub_post = self.last_model.posterior(sub_eval)
                pred_means_list.append(sub_post.mean.view(n_sub, num_tasks))
            pred_means = torch.cat(pred_means_list, dim=0)

        # 1. Baseline default candidate selection from predicted ensemble average
        avg_pred_means = pred_means.mean(dim=-1)
        best_cand_def_idx = int(torch.argmin(avg_pred_means).item())
        cand_def_theta = unique_thetas_np[best_cand_def_idx]
        u_def_seed = self._normalize(cand_def_theta)

        # Polish the Universal Default candidate continuously on the surrogate ensemble mean
        for param in self.last_model.parameters():
            param.requires_grad = False

        def polish_surrogate_default(u_seed: np.ndarray, delta: float = 0.10) -> np.ndarray:
            bounds = [(max(0.0, float(u_seed[i] - delta)), min(1.0, float(u_seed[i] + delta))) for i in range(len(u_seed))]
            def cost_def(u_np: np.ndarray) -> tuple[float, np.ndarray]:
                u_t = torch.tensor(u_np, dtype=torch.double, requires_grad=True)
                post = self.last_model.posterior(u_t.unsqueeze(0).unsqueeze(1))
                loss = post.mean.squeeze(0).squeeze(0).mean()
                grad_tensor = torch.autograd.grad(loss, u_t, retain_graph=False)[0]
                grad = grad_tensor.detach().cpu().numpy()
                return float(loss.item()), grad

            res = minimize(cost_def, u_seed, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 20})
            return res.x if res.success or res.x is not None else u_seed

        u_def_cont = polish_surrogate_default(u_def_seed)
        theta_def_cont = self._unnormalize(u_def_cont)

        # Ground truth evaluation of polished default across all tasks
        y_def_tasks = self.evaluate_ensemble(theta_def_cont)
        best_def_loss = float(y_def_tasks.mean())
        self.best_theta_def = theta_def_cont

        # 2. Polish each individual task continuously on the surrogate
        def polish_surrogate_task(task_idx: int, u_seed: np.ndarray, delta: float = 0.10) -> np.ndarray:
            bounds = [(max(0.0, float(u_seed[i] - delta)), min(1.0, float(u_seed[i] + delta))) for i in range(len(u_seed))]
            def cost_task(u_np: np.ndarray) -> tuple[float, np.ndarray]:
                u_t = torch.tensor(u_np, dtype=torch.double, requires_grad=True)
                post = self.last_model.posterior(u_t.unsqueeze(0).unsqueeze(1))
                loss = post.mean.squeeze(0).squeeze(0)[task_idx]
                grad_tensor = torch.autograd.grad(loss, u_t, retain_graph=False)[0]
                grad = grad_tensor.detach().cpu().numpy()
                return float(loss.item()), grad

            res = minimize(cost_task, u_seed, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 20})
            return res.x if res.success or res.x is not None else u_seed

        polished_thetas: list[np.ndarray] = []
        polish_requests: list[tuple[np.ndarray, int]] = []
        best_bo_thetas: list[np.ndarray] = []
        best_bo_vals: list[float] = []

        for j in range(num_tasks):
            hist_j = history_by_game[j]
            best_th_bo_j, best_y_bo_j = min(hist_j, key=lambda pt: pt[1])
            best_bo_thetas.append(best_th_bo_j)
            best_bo_vals.append(best_y_bo_j)

            u_seed_j = self._normalize(best_th_bo_j)
            u_cont_j = polish_surrogate_task(j, u_seed_j)
            th_cont_j = self._unnormalize(u_cont_j)
            polished_thetas.append(th_cont_j)
            polish_requests.append((th_cont_j, j))

        # Single parallel verification batch for all continuous tailored candidates
        y_cont_tasks = self.evaluate_batch(polish_requests)

        # 3. Three-way comparison for each task
        self.best_thetas = []
        best_tailored_vals: list[float] = []
        self.anomaly_count = 0
        self.refinements_executed = num_tasks

        count_continuous = 0
        count_bo = 0
        count_default = 0

        for j in range(num_tasks):
            loss_cont = float(y_cont_tasks[j])
            loss_bo = float(best_bo_vals[j])
            loss_def = float(y_def_tasks[j])

            th_cont = polished_thetas[j]
            th_bo = best_bo_thetas[j]
            th_def = theta_def_cont

            # Determine best candidate
            tol = 1e-8
            if loss_cont <= loss_bo + tol and loss_cont <= loss_def + tol:
                chosen_th = th_cont
                chosen_val = loss_cont
                count_continuous += 1
            elif loss_bo <= loss_cont + tol and loss_bo <= loss_def + tol:
                chosen_th = th_bo
                chosen_val = loss_bo
                count_bo += 1
            else:
                chosen_th = th_def
                chosen_val = loss_def
                count_default += 1
                self.anomaly_count += 1
                warnings.warn(
                    f"[MTBO Anomaly] Game {j}: Universal default outperformed both continuous and discrete tailored optima "
                    f"(y_def={loss_def:.5f} < min(y_cont={loss_cont:.5f}, y_bo={loss_bo:.5f})).",
                    UserWarning,
                )

            self.best_thetas.append(chosen_th)
            best_tailored_vals.append(chosen_val)

        # Determine overall winning optima across ensemble
        avg_loss_cont = float(np.mean(y_cont_tasks))
        avg_loss_bo = float(np.mean(best_bo_vals))
        avg_loss_def = float(best_def_loss)

        if avg_loss_cont <= avg_loss_bo + 1e-8 and avg_loss_cont <= avg_loss_def + 1e-8:
            overall_winner = "Continuous exploration (Surrogate Polish)"
        elif avg_loss_bo <= avg_loss_cont + 1e-8 and avg_loss_bo <= avg_loss_def + 1e-8:
            overall_winner = "Individual task optima from discrete BO"
        else:
            overall_winner = "Default optima"
            warnings.warn(
                f"[MTBO Anomaly] Universal default optima outperformed tailored optima on average "
                f"(avg_loss_def={avg_loss_def:.5f} < min(avg_loss_cont={avg_loss_cont:.5f}, avg_loss_bo={avg_loss_bo:.5f})).",
                UserWarning,
            )

        # Print summary report
        pct_cont = (count_continuous / num_tasks) * 100.0
        pct_bo = (count_bo / num_tasks) * 100.0
        pct_def = (count_default / num_tasks) * 100.0
        print(
            f"\n=== MTBO Post-Evaluation Polish Summary ===\n"
            f"Total Tasks: {num_tasks}\n"
            f"1. Continuous exploration:       {count_continuous:3d} ({pct_cont:5.1f}%) | Avg Loss: {avg_loss_cont:.5f}\n"
            f"2. Individual task (Discrete BO): {count_bo:3d} ({pct_bo:5.1f}%) | Avg Loss: {avg_loss_bo:.5f}\n"
            f"3. Default optima:                {count_default:3d} ({pct_def:5.1f}%) | Avg Loss: {avg_loss_def:.5f}\n"
            f"-------------------------------------------\n"
            f"BEST OPTIMA AFTER POLISH: {overall_winner}\n"
            f"===========================================",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        for j in range(num_tasks):
            opt_dict_j = self._flatten_hyperparams(self.best_thetas[j], self.game_ensemble[j])
            opt_dict_j["game_id"] = f"game_{j}"
            opt_dict_j["is_default"] = False
            opt_dict_j["objective_value"] = best_tailored_vals[j]
            results.append(opt_dict_j)

        opt_dict_def = self._flatten_hyperparams(self.best_theta_def, self.game_ensemble[0])
        opt_dict_def["game_id"] = "default"
        opt_dict_def["is_default"] = True
        opt_dict_def["objective_value"] = best_def_loss
        results.append(opt_dict_def)

        if self.output_path is not None:
            self._save_csv(results)

        return results

    def run_parallel_single_task_bo(
        self,
        n_init: Optional[int] = None,
        max_bo_steps: Optional[int] = None,
        min_bo_steps: Optional[int] = None,
        patience: Optional[int] = None,
        eps_ei: Optional[float] = None,
        tol_stagnation: Optional[float] = None,
        sim_per_param: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        if sim_per_param is not None:
            self.sim_per_param = max(1, sim_per_param)

        n_init = self.n_init if n_init is None else max(1, n_init)
        max_bo_steps = self.max_bo_steps if max_bo_steps is None else max(1, max_bo_steps)
        min_bo_steps = self.min_bo_steps if min_bo_steps is None else max(0, min_bo_steps)
        patience = self.patience if patience is None else max(1, patience)
        eps_ei = self.eps_ei if eps_ei is None else eps_ei
        tol_stagnation = self.tol_stagnation if tol_stagnation is None else tol_stagnation

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        num_tasks = self.n_tasks
        unit_bounds = torch.tensor([[0.0] * self.dim, [1.0] * self.dim], dtype=torch.double)

        task_train_x: dict[int, torch.Tensor] = {}
        task_train_y: dict[int, torch.Tensor] = {}
        history_by_game: dict[int, list[tuple[np.ndarray, float]]] = {j: [] for j in range(num_tasks)}
        best_tailored_history: dict[int, list[float]] = {j: [] for j in range(num_tasks)}

        init_requests: list[tuple[np.ndarray, int]] = []
        task_init_thetas: dict[int, list[np.ndarray]] = {j: [] for j in range(num_tasks)}
        task_init_units: dict[int, list[np.ndarray]] = {j: [] for j in range(num_tasks)}

        for j in range(num_tasks):
            task_seed = (self.seed + j) if self.seed is not None else None
            sobol = torch.quasirandom.SobolEngine(dimension=self.dim, scramble=True, seed=task_seed)
            u_samples = sobol.draw(n_init).to(dtype=torch.double).numpy()
            for u in u_samples:
                th = self._unnormalize(u)
                task_init_units[j].append(u)
                task_init_thetas[j].append(th)
                init_requests.append((th, j))

        init_losses = self.evaluate_batch(init_requests)
        loss_idx = 0
        for j in range(num_tasks):
            x_rows = []
            y_rows = []
            for u, th in zip(task_init_units[j], task_init_thetas[j]):
                loss_val = init_losses[loss_idx]
                loss_idx += 1
                history_by_game[j].append((th, loss_val))
                x_rows.append(torch.tensor(u, dtype=torch.double).unsqueeze(0))
                y_rows.append(torch.tensor([[loss_val]], dtype=torch.double))
            task_train_x[j] = torch.cat(x_rows, dim=0)
            task_train_y[j] = torch.cat(y_rows, dim=0)

        active_tasks = set(range(num_tasks))

        for step in range(max_bo_steps):
            cands_ind: dict[int, np.ndarray] = {}
            eis_ind: dict[int, float] = {}

            for j in sorted(active_tasks):
                model_j = SingleTaskGP(task_train_x[j], task_train_y[j], outcome_transform=Standardize(m=1))
                mll_j = ExactMarginalLogLikelihood(model_j.likelihood, model_j)
                fit_gpytorch_mll(mll_j, optimizer_kwargs={"options": {"maxiter": 25}})

                best_f_j = float(task_train_y[j].min().item())
                acq_j = LogExpectedImprovement(model=model_j, best_f=best_f_j, maximize=False)
                cand, acq_val = optimize_acqf(
                    acq_function=acq_j,
                    bounds=unit_bounds,
                    q=1,
                    num_restarts=8,
                    raw_samples=128,
                )
                cands_ind[j] = cand.squeeze(0).detach().cpu().numpy()
                eis_ind[j] = float(np.exp(min(50.0, acq_val.item())))
                del model_j, mll_j, acq_j

            tasks_to_prune: set[int] = set()
            for j in sorted(active_tasks):
                best_val_j = min([pt[1] for pt in history_by_game[j]])
                best_tailored_history[j].append(best_val_j)
                pruned_ei = (eps_ei is not None and eis_ind[j] < eps_ei)
                pruned_stagnation = False
                imp_j = 0.0
                if len(best_tailored_history[j]) >= patience:
                    imp_j = best_tailored_history[j][-patience] - best_tailored_history[j][-1]
                    pruned_stagnation = (imp_j < tol_stagnation)
                if (step + 1 >= min_bo_steps) and (pruned_ei or pruned_stagnation):
                    tasks_to_prune.add(j)

            for j in tasks_to_prune:
                active_tasks.remove(j)

            if len(active_tasks) == 0:
                break

            step_requests = [(self._unnormalize(cands_ind[j]), j) for j in sorted(active_tasks)]
            step_losses = self.evaluate_batch(step_requests)

            for req_idx, j in enumerate(sorted(active_tasks)):
                cand_u = cands_ind[j]
                th_j = self._unnormalize(cand_u)
                loss_j = step_losses[req_idx]
                history_by_game[j].append((th_j, loss_j))
                task_train_x[j] = torch.cat([task_train_x[j], torch.tensor(cand_u, dtype=torch.double).unsqueeze(0)], dim=0)
                task_train_y[j] = torch.cat([task_train_y[j], torch.tensor([[loss_j]], dtype=torch.double)], dim=0)

            self._garbage_collect()

        self.best_thetas = []
        best_tailored_vals: list[float] = []
        polished_thetas: list[np.ndarray] = []
        polish_requests: list[tuple[np.ndarray, int]] = []
        best_bo_thetas: list[np.ndarray] = []
        best_bo_vals: list[float] = []

        for j in range(num_tasks):
            best_th_bo_j, best_y_bo_j = min(history_by_game[j], key=lambda pt: pt[1])
            best_bo_thetas.append(best_th_bo_j)
            best_bo_vals.append(best_y_bo_j)

            u_seed_j = self._normalize(best_th_bo_j)
            model_j = SingleTaskGP(task_train_x[j], task_train_y[j], outcome_transform=Standardize(m=1))
            mll_j = ExactMarginalLogLikelihood(model_j.likelihood, model_j)
            fit_gpytorch_mll(mll_j, optimizer_kwargs={"options": {"maxiter": 20}})

            for param in model_j.parameters():
                param.requires_grad = False

            def polish_task(u_seed: np.ndarray, mod: SingleTaskGP, delta: float = 0.10) -> np.ndarray:
                bounds = [(max(0.0, float(u_seed[i] - delta)), min(1.0, float(u_seed[i] + delta))) for i in range(len(u_seed))]
                def cost_task(u_np: np.ndarray) -> tuple[float, np.ndarray]:
                    u_t = torch.tensor(u_np, dtype=torch.double, requires_grad=True)
                    post = mod.posterior(u_t.unsqueeze(0))
                    loss = post.mean.squeeze(0).squeeze(0)
                    grad_tensor = torch.autograd.grad(loss, u_t, retain_graph=False)[0]
                    grad = grad_tensor.detach().cpu().numpy()
                    return float(loss.item()), grad
                res = minimize(cost_task, u_seed, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 20})
                return res.x if res.success or res.x is not None else u_seed

            u_cont_j = polish_task(u_seed_j, model_j)
            th_cont_j = self._unnormalize(u_cont_j)
            polished_thetas.append(th_cont_j)
            polish_requests.append((th_cont_j, j))
            del model_j, mll_j

        self._garbage_collect()
        y_cont_tasks = self.evaluate_batch(polish_requests)

        results: list[dict[str, Any]] = []
        for j in range(num_tasks):
            loss_cont = float(y_cont_tasks[j])
            loss_bo = float(best_bo_vals[j])
            if loss_cont <= loss_bo + 1e-8:
                chosen_th = polished_thetas[j]
                chosen_val = loss_cont
            else:
                chosen_th = best_bo_thetas[j]
                chosen_val = loss_bo

            self.best_thetas.append(chosen_th)
            best_tailored_vals.append(chosen_val)

            opt_dict_j = self._flatten_hyperparams(chosen_th, self.game_ensemble[j])
            opt_dict_j["game_id"] = f"game_{j}"
            opt_dict_j["is_default"] = False
            opt_dict_j["objective_value"] = chosen_val
            results.append(opt_dict_j)

        if self.output_path is not None:
            self._save_csv(results)

        self._garbage_collect()
        return results

    def validate(self) -> dict[str, Any]:
        if self.train_x is None or self.train_y is None or self.last_model is None:
            raise RuntimeError("Cannot validate before running run_optimization_loop().")

        model = self.last_model
        with torch.no_grad():
            train_x = model.train_inputs[0]
            model.train()
            mvn = model(train_x)
            lik_dist = model.likelihood(mvn, train_x)
            covar_full = lik_dist.covariance_matrix
            model.eval()

            K_full = covar_full.to_dense() if hasattr(covar_full, "to_dense") else covar_full
            reg = 1e-6 * torch.eye(K_full.shape[0], dtype=torch.double, device=K_full.device)
            K_inv = torch.linalg.inv(K_full + reg)

            y = model.train_targets.view(-1, 1)
            alpha = K_inv @ y
            diag_inv = torch.diagonal(K_inv).view(-1, 1)

            mu_loo = y - alpha / diag_inv
            var_loo = torch.clamp(1.0 / diag_inv, min=1e-8)

            errors = y - mu_loo
            loo_rmse = float(torch.sqrt(torch.mean(errors ** 2)).item())
            var_y = float(torch.var(y).item())
            loo_smse = float((torch.mean(errors ** 2) / (var_y + 1e-12)).item())
            z_scores = errors / torch.sqrt(var_loo)
            cal_std = float(torch.std(z_scores).item())

            subopt_bounds: dict[str, float] = {}
            for j in range(self.n_tasks):
                subopt_bounds[f"game_{j}"] = max(0.0, float(loo_rmse))

        metrics = {
            "loo_rmse": loo_rmse,
            "loo_smse": loo_smse,
            "calibration_std": cal_std,
            "suboptimality_bounds": subopt_bounds,
            "anomalies_detected": self.anomaly_count,
            "refinements_executed": self.refinements_executed,
        }
        self.validation_metrics = metrics
        return metrics

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
