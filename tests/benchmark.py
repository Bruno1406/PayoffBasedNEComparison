import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import optuna
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.objective import LinearMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import IIDNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from payoffnash import NESeeking, QuadraticGame, generate_quadratic_games
except ImportError:
    from ..payoffnash import NESeeking, QuadraticGame, generate_quadratic_games

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAM_BOUNDS = {
    "k[0,0]": (0.01, 10.0),
    "k[1,0]": (0.01, 10.0),
    "a[0,0]": (0.01, 5.0),
    "a[1,0]": (0.01, 5.0),
    "omega[0,0]": (10.0, 50.0),
    "omega[1,0]": (10.0, 50.0),
    "omega_h[0]": (1.0, 20.0),
    "omega_h[1]": (1.0, 20.0),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())
DIM = len(PARAM_NAMES)
LOG_LOWER = np.array([np.log10(PARAM_BOUNDS[k][0]) for k in PARAM_NAMES], dtype=float)
LOG_UPPER = np.array([np.log10(PARAM_BOUNDS[k][1]) for k in PARAM_NAMES], dtype=float)


def unit_to_theta(unit_x: np.ndarray) -> np.ndarray:
    log_vals = LOG_LOWER + (LOG_UPPER - LOG_LOWER) * unit_x
    return 10.0 ** log_vals


def theta_to_unit(theta: np.ndarray) -> np.ndarray:
    log_vals = np.log10(np.clip(theta, 1e-6, None))
    return np.clip((log_vals - LOG_LOWER) / (LOG_UPPER - LOG_LOWER), 0.0, 1.0)


def quadratic_game_regret_cost(game: QuadraticGame, history, dt: float = 0.01) -> float:
    nash_eq = game.nash_equilibrium.ravel()
    actions = history.action_history
    dev_p1 = np.column_stack([nash_eq[0] * np.ones(len(actions)), actions[:, 1]])
    payoff_p1_dev = np.array([game.payoff_functions[0](*a) for a in dev_p1])
    regret_p1 = np.maximum(0.0, payoff_p1_dev - history.payoff_history[:, 0])

    dev_p2 = np.column_stack([actions[:, 0], nash_eq[1] * np.ones(len(actions))])
    payoff_p2_dev = np.array([game.payoff_functions[1](*a) for a in dev_p2])
    regret_p2 = np.maximum(0.0, payoff_p2_dev - history.payoff_history[:, 1])

    total_regret = np.sum(regret_p1 + regret_p2) * dt
    return float(np.log10(np.maximum(total_regret, 1e-12)))


def evaluate_game(game: QuadraticGame, theta: np.ndarray, num_steps: int = 2000, dt: float = 0.01, seed: int = 123) -> float:
    np.random.seed(seed)
    k = np.array([[theta[0]], [theta[1]]])
    a = np.array([[theta[2]], [theta[3]]])
    omega = np.array([[theta[4]], [theta[5]]])
    omega_h = np.array([theta[6], theta[7]])
    est = NESeeking(game, k=k, a=a, omega=omega, omega_h=omega_h, dt=dt)
    hist = est.estimate(np.zeros((game.N, 1)), num_steps=num_steps)
    return quadratic_game_regret_cost(game, hist, dt=dt)


def evaluate_ensemble(games: list[QuadraticGame], theta: np.ndarray, num_steps: int = 2000, dt: float = 0.01, seed: int = 123) -> list[float]:
    return [evaluate_game(g, theta, num_steps=num_steps, dt=dt, seed=seed) for g in games]


@dataclass
class BenchmarkRunResult:
    method: str
    ensemble_size: int
    wall_clock_time: float
    total_evaluations: int
    num_bo_steps: int
    ensemble_mean_loss_default: float
    mean_tailored_loss: float
    theta_default: np.ndarray
    tailored_thetas: list[np.ndarray]
    tailored_losses: list[float]
    tunabilities: list[float]
    anomaly_count: int
    learned_task_covar: Optional[np.ndarray] = None


class ToleranceStoppingCallback:
    def __init__(self, patience: int = 15, tol: float = 1e-4):
        self.patience = patience
        self.tol = tol
        self.best_value = float("inf")
        self.no_improve_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        if trial.value < self.best_value - self.tol:
            self.best_value = trial.value
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            if self.no_improve_count >= self.patience:
                study.stop()


def run_optuna_baseline(
    games: list[QuadraticGame],
    num_steps: int = 2000,
    max_trials_per_game: int = 50,
    patience: int = 15,
    seed: int = 42,
) -> BenchmarkRunResult:
    m_tasks = len(games)
    t0 = time.time()
    tailored_thetas: list[np.ndarray] = []
    tailored_losses: list[float] = []
    all_evaluated_thetas: list[np.ndarray] = []
    total_evals = 0

    for j, game in enumerate(games):
        cb = ToleranceStoppingCallback(patience=patience, tol=1e-4)
        sampler = optuna.samplers.TPESampler(seed=seed + j)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            vals = [trial.suggest_float(p, PARAM_BOUNDS[p][0], PARAM_BOUNDS[p][1], log=True) for p in PARAM_NAMES]
            theta = np.array(vals, dtype=float)
            all_evaluated_thetas.append(theta)
            return evaluate_game(game, theta, num_steps=num_steps, seed=seed)

        study.optimize(objective, n_trials=max_trials_per_game, callbacks=[cb])
        total_evals += len(study.trials)

        best_theta = np.array([study.best_params[p] for p in PARAM_NAMES], dtype=float)
        tailored_thetas.append(best_theta)
        tailored_losses.append(study.best_value)

    # Determine default theta from candidates evaluated across the ensemble
    unique_candidates = []
    seen = set()
    for th in tailored_thetas + all_evaluated_thetas[:min(len(all_evaluated_thetas), 30)]:
        key = tuple(np.round(th, 4))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(th)

    best_default_mean = float("inf")
    best_theta_def = unique_candidates[0]
    for cand in unique_candidates:
        losses = evaluate_ensemble(games, cand, num_steps=num_steps, seed=seed)
        total_evals += m_tasks
        avg_loss = float(np.mean(losses))
        if avg_loss < best_default_mean:
            best_default_mean = avg_loss
            best_theta_def = cand

    default_losses = evaluate_ensemble(games, best_theta_def, num_steps=num_steps, seed=seed)
    tunabilities = [default_losses[j] - tailored_losses[j] for j in range(m_tasks)]

    wall_time = time.time() - t0
    return BenchmarkRunResult(
        method="Optuna_Baseline",
        ensemble_size=m_tasks,
        wall_clock_time=wall_time,
        total_evaluations=total_evals,
        num_bo_steps=max_trials_per_game,
        ensemble_mean_loss_default=best_default_mean,
        mean_tailored_loss=float(np.mean(tailored_losses)),
        theta_default=best_theta_def,
        tailored_thetas=tailored_thetas,
        tailored_losses=tailored_losses,
        tunabilities=tunabilities,
        anomaly_count=0,
    )


def run_mtbo_multi_acquisition(
    games: list[QuadraticGame],
    approach: str = "A",  # 'A' (2M) or 'B' (M+1)
    num_steps: int = 2000,
    n_init: int = 10,
    max_bo_steps: int = 30,
    eps_ei: float = 1e-4,
    ei_window: int = 3,
    seed: int = 42,
) -> BenchmarkRunResult:
    m_tasks = len(games)
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Quasi-random Sobol initialization in [0, 1]^d
    sobol_eng = torch.quasirandom.SobolEngine(dimension=DIM, scramble=True, seed=seed)
    init_units = sobol_eng.draw(n_init).to(dtype=torch.double).numpy()

    train_x_list: list[torch.Tensor] = []
    train_y_list: list[torch.Tensor] = []
    total_evals = 0

    history_by_game: dict[int, list[tuple[np.ndarray, float]]] = {j: [] for j in range(m_tasks)}

    for u in init_units:
        th = unit_to_theta(u)
        losses = evaluate_ensemble(games, th, num_steps=num_steps, seed=seed)
        total_evals += m_tasks
        for j, loss in enumerate(losses):
            history_by_game[j].append((th, loss))
            row_x = torch.cat([torch.tensor(u, dtype=torch.double), torch.tensor([float(j)], dtype=torch.double)])
            train_x_list.append(row_x.unsqueeze(0))
            train_y_list.append(torch.tensor([[loss]], dtype=torch.double))

    train_x = torch.cat(train_x_list, dim=0)
    train_y = torch.cat(train_y_list, dim=0)

    bounds = torch.tensor([[0.0] * DIM, [1.0] * DIM], dtype=torch.double)
    active_tasks = set(range(m_tasks))
    avg_ei_below_count = 0
    step_count = 0
    prev_state_dict: Optional[dict[str, Any]] = None
    learned_covar: Optional[np.ndarray] = None

    for step in range(1, max_bo_steps + 1):
        step_count = step
        # Instantiate MultiTaskGP and fit MLL
        model = MultiTaskGP(train_x, train_y, task_feature=-1, outcome_transform=Standardize(m=1))
        if prev_state_dict is not None:
            try:
                model.load_state_dict(prev_state_dict, strict=False)
            except Exception:
                pass

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, optimizer_kwargs={"options": {"maxiter": 25}})
        prev_state_dict = model.state_dict()

        if hasattr(model, "covar_module") and hasattr(model.covar_module, "kernels") and len(model.covar_module.kernels) > 1:
            try:
                learned_covar = model.covar_module.kernels[1].covar_matrix.detach().cpu().numpy()
            except Exception:
                pass

        sampler = IIDNormalSampler(sample_shape=torch.Size([64]), seed=seed + step)

        # 1. Average acquisition function
        w_avg = torch.full((m_tasks,), -1.0 / m_tasks, dtype=torch.double)
        obj_avg = LinearMCObjective(weights=w_avg)

        # best_f for average
        shared_avg_losses = []
        for i in range(len(train_x) // m_tasks):
            block_y = train_y[i * m_tasks : (i + 1) * m_tasks]
            shared_avg_losses.append(-float(block_y.mean().item()))
        best_f_avg = torch.tensor(max(shared_avg_losses) if shared_avg_losses else 0.0, dtype=torch.double)

        acqf_avg = qLogExpectedImprovement(model=model, best_f=best_f_avg, sampler=sampler, objective=obj_avg)
        cand_avg_unit, val_avg = optimize_acqf(
            acq_function=acqf_avg,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
            options={"maxiter": 25},
        )
        ei_avg = float(torch.exp(val_avg).item())

        # 2. Individual acquisition functions for active tasks
        cands_ind: dict[int, np.ndarray] = {}
        eis_ind: dict[int, float] = {}
        for j in sorted(active_tasks):
            w_j = torch.zeros(m_tasks, dtype=torch.double)
            w_j[j] = -1.0
            obj_j = LinearMCObjective(weights=w_j)
            best_val_j = min([pt[1] for pt in history_by_game[j]])
            best_f_j = torch.tensor(-best_val_j, dtype=torch.double)

            acqf_j = qLogExpectedImprovement(model=model, best_f=best_f_j, sampler=sampler, objective=obj_j)
            cand_j_unit, val_j = optimize_acqf(
                acq_function=acqf_j,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=64,
                options={"maxiter": 25},
            )
            ei_j = float(torch.exp(val_j).item())
            eis_ind[j] = ei_j
            cands_ind[j] = cand_j_unit.squeeze(0).detach().cpu().numpy()

        # Update active tasks pruning
        for j, ei_val in eis_ind.items():
            if ei_val < eps_ei:
                active_tasks.remove(j)

        if ei_avg < eps_ei:
            avg_ei_below_count += 1
        else:
            avg_ei_below_count = 0

        if step % 5 == 1 or step == max_bo_steps or len(active_tasks) == 0:
            print(f"    [BO Step {step:2d}/{max_bo_steps}] qEI_avg: {ei_avg:.2e} | Active tasks: {len(active_tasks)}/{m_tasks} | Best Shared Avg: {-best_f_avg.item():.4f}")

        # Check early stopping: average converged AND all tasks pruned
        if avg_ei_below_count >= ei_window and len(active_tasks) == 0:
            print(f"    [BO Early Stop at Step {step}] EI convergence reached across average and all tasks.")
            break

        cand_avg_np = cand_avg_unit.squeeze(0).detach().cpu().numpy()
        theta_avg = unit_to_theta(cand_avg_np)

        # Execute evaluations according to Approach A vs Approach B
        new_x_step: list[torch.Tensor] = []
        new_y_step: list[torch.Tensor] = []

        if approach.upper() == "A":
            # 2M approach: evaluate theta_avg across all M games
            losses_avg = evaluate_ensemble(games, theta_avg, num_steps=num_steps, seed=seed)
            total_evals += m_tasks
            for j, loss in enumerate(losses_avg):
                history_by_game[j].append((theta_avg, loss))
                row_x = torch.cat([torch.tensor(cand_avg_np, dtype=torch.double), torch.tensor([float(j)], dtype=torch.double)])
                new_x_step.append(row_x.unsqueeze(0))
                new_y_step.append(torch.tensor([[loss]], dtype=torch.double))

        else:
            # M+1 approach: evaluate theta_avg on single best task t*
            # Query model posterior at theta_avg across tasks to find maximum single-task improvement/uncertainty
            with torch.no_grad():
                post_avg = model.posterior(cand_avg_unit.view(1, 1, DIM))
                post_mean = post_avg.mean.view(-1)
                post_var = post_avg.variance.view(-1)

            best_t = 0
            best_ei_t = -float("inf")
            for t in range(m_tasks):
                y_best_t = min([pt[1] for pt in history_by_game[t]])
                sigma_t = float(torch.sqrt(post_var[t]).item()) + 1e-8
                delta_t = y_best_t - float(post_mean[t].item())
                z = delta_t / sigma_t
                ei_t = delta_t * norm.cdf(z) + sigma_t * norm.pdf(z)
                if ei_t > best_ei_t:
                    best_ei_t = ei_t
                    best_t = t

            loss_t = evaluate_game(games[best_t], theta_avg, num_steps=num_steps, seed=seed)
            total_evals += 1
            history_by_game[best_t].append((theta_avg, loss_t))
            row_x = torch.cat([torch.tensor(cand_avg_np, dtype=torch.double), torch.tensor([float(best_t)], dtype=torch.double)])
            new_x_step.append(row_x.unsqueeze(0))
            new_y_step.append(torch.tensor([[loss_t]], dtype=torch.double))

        # Evaluate active individual tasks
        for j in list(cands_ind.keys()):
            if j in active_tasks:
                cand_u = cands_ind[j]
                th_j = unit_to_theta(cand_u)
                loss_j = evaluate_game(games[j], th_j, num_steps=num_steps, seed=seed)
                total_evals += 1
                history_by_game[j].append((th_j, loss_j))
                row_x = torch.cat([torch.tensor(cand_u, dtype=torch.double), torch.tensor([float(j)], dtype=torch.double)])
                new_x_step.append(row_x.unsqueeze(0))
                new_y_step.append(torch.tensor([[loss_j]], dtype=torch.double))

        if new_x_step:
            train_x = torch.cat([train_x] + new_x_step, dim=0)
            train_y = torch.cat([train_y] + new_y_step, dim=0)

    # Post-optimization extraction & anomaly detection
    # Default theta: pick evaluated theta minimizing ensemble average
    all_evaluated_thetas = []
    for j in range(m_tasks):
        all_evaluated_thetas.extend([pt[0] for pt in history_by_game[j]])

    unique_cands = []
    seen = set()
    for th in all_evaluated_thetas:
        k_tpl = tuple(np.round(th, 4))
        if k_tpl not in seen:
            seen.add(k_tpl)
            unique_cands.append(th)

    best_def_loss = float("inf")
    best_theta_def = unique_cands[0]
    for cand in unique_cands[:min(len(unique_cands), 30)]:
        losses = evaluate_ensemble(games, cand, num_steps=num_steps, seed=seed)
        total_evals += m_tasks
        mean_l = float(np.mean(losses))
        if mean_l < best_def_loss:
            best_def_loss = mean_l
            best_theta_def = cand

    default_losses_on_games = evaluate_ensemble(games, best_theta_def, num_steps=num_steps, seed=seed)

    tailored_thetas: list[np.ndarray] = []
    tailored_losses: list[float] = []
    tunabilities: list[float] = []
    anomaly_count = 0

    for j in range(m_tasks):
        pts = history_by_game[j]
        pts_sorted = sorted(pts, key=lambda x: x[1])
        best_th_j, best_val_j = pts_sorted[0]
        y_def_j = default_losses_on_games[j]

        if y_def_j < best_val_j:
            warnings.warn(
                f"[MTBO Anomaly] Game {j}: Default theta performed better than tailored theta "
                f"(y_def={y_def_j:.4f} < y_tailored={best_val_j:.4f}). Applying non-negative safeguard.",
                UserWarning,
            )
            anomaly_count += 1
            best_val_j = y_def_j
            best_th_j = best_theta_def

        tailored_thetas.append(best_th_j)
        tailored_losses.append(best_val_j)
        tunabilities.append(y_def_j - best_val_j)

    wall_time = time.time() - t0
    return BenchmarkRunResult(
        method=f"MTBO_Approach_{approach}",
        ensemble_size=m_tasks,
        wall_clock_time=wall_time,
        total_evaluations=total_evals,
        num_bo_steps=step_count,
        ensemble_mean_loss_default=best_def_loss,
        mean_tailored_loss=float(np.mean(tailored_losses)),
        theta_default=best_theta_def,
        tailored_thetas=tailored_thetas,
        tailored_losses=tailored_losses,
        tunabilities=tunabilities,
        anomaly_count=anomaly_count,
        learned_task_covar=learned_covar,
    )


def print_comparison_table(results: list[BenchmarkRunResult]) -> None:
    print("\n" + "=" * 105)
    print(f"{'Method':<20} | {'M':<3} | {'Steps':<6} | {'Evals':<6} | {'Time (s)':<9} | {'Def Loss':<10} | {'Tailored Loss':<13} | {'Anomalies':<9}")
    print("-" * 105)
    for r in results:
        print(
            f"{r.method:<20} | {r.ensemble_size:<3} | {r.num_bo_steps:<6} | {r.total_evaluations:<6} | "
            f"{r.wall_clock_time:<9.2f} | {r.ensemble_mean_loss_default:<10.4f} | {r.mean_tailored_loss:<13.4f} | {r.anomaly_count:<9}"
        )
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Acquisition MTBO Benchmark vs Optuna on NESeeking (d=8)")
    parser.add_argument("--ensemble-sizes", nargs="+", type=int, default=[1, 5, 10, 20], help="Ensemble sizes M to test")
    parser.add_argument("--num-sim-steps", type=int, default=2000, help="Simulation steps per game evaluation")
    parser.add_argument("--n-init", type=int, default=10, help="Initial quasi-random points")
    parser.add_argument("--max-bo-steps", type=int, default=25, help="Maximum sequential BO steps")
    parser.add_argument("--quick-test", action="store_true", help="Run fast verification smoke test")
    args = parser.parse_args()

    if args.quick_test:
        print("=== Running Quick Verification Smoke Test (M=1, M=2, 3 BO steps) ===")
        sizes = [1, 2]
        max_steps = 3
        sim_steps = 500
        n_init = 4
    else:
        sizes = args.ensemble_sizes
        max_steps = args.max_bo_steps
        sim_steps = args.num_sim_steps
        n_init = args.n_init

    all_results: list[BenchmarkRunResult] = []

    for m in sizes:
        print(f"\n=======================================================")
        print(f"BENCHMARKING ENSEMBLE SIZE M = {m} (NESeeking, d={DIM})")
        print(f"=======================================================")
        games = generate_quadratic_games(ensemble_size=m, N=2, m=1, alpha=0.05, seed=42)

        # 1. Approach A (2M evals)
        print(f"\n[1/3] Running MTBO Approach A (2M with Active Pruning)...")
        res_a = run_mtbo_multi_acquisition(
            games=games,
            approach="A",
            num_steps=sim_steps,
            n_init=n_init,
            max_bo_steps=max_steps,
            eps_ei=1e-4,
        )
        all_results.append(res_a)
        print(f"Approach A finished in {res_a.wall_clock_time:.2f}s ({res_a.total_evaluations} evals, def_loss={res_a.ensemble_mean_loss_default:.4f})")

        # 2. Approach B (M+1 evals)
        print(f"\n[2/3] Running MTBO Approach B (M+1 with Active Pruning)...")
        res_b = run_mtbo_multi_acquisition(
            games=games,
            approach="B",
            num_steps=sim_steps,
            n_init=n_init,
            max_bo_steps=max_steps,
            eps_ei=1e-4,
        )
        all_results.append(res_b)
        print(f"Approach B finished in {res_b.wall_clock_time:.2f}s ({res_b.total_evaluations} evals, def_loss={res_b.ensemble_mean_loss_default:.4f})")

        # 3. Fair Optuna Baseline
        print(f"\n[3/3] Running Fair Optuna Baseline (Per-game TPE)...")
        res_opt = run_optuna_baseline(
            games=games,
            num_steps=sim_steps,
            max_trials_per_game=max_steps * 2 if not args.quick_test else 5,
            patience=10 if not args.quick_test else 3,
        )
        all_results.append(res_opt)
        print(f"Optuna Baseline finished in {res_opt.wall_clock_time:.2f}s ({res_opt.total_evaluations} evals, def_loss={res_opt.ensemble_mean_loss_default:.4f})")

    print_comparison_table(all_results)

    # Save to data/04_mtbo_multi_acquisition_benchmark/
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "04_mtbo_multi_acquisition_benchmark"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    md_path = os.path.join(out_dir, "benchmark_report.md")

    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "ensemble_size", "bo_steps", "total_evals", "wall_clock_time", "default_loss", "mean_tailored_loss", "anomalies"])
        for r in all_results:
            writer.writerow([r.method, r.ensemble_size, r.num_bo_steps, r.total_evaluations, f"{r.wall_clock_time:.2f}", f"{r.ensemble_mean_loss_default:.4f}", f"{r.mean_tailored_loss:.4f}", r.anomaly_count])

    with open(md_path, "w") as f:
        f.write("# MTBO Multi-Acquisition (Approach A vs B vs Optuna) Benchmark Report\n\n")
        f.write("| Method | M | Steps | Total Evals | Time (s) | Default Loss | Tailored Loss | Anomalies |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in all_results:
            f.write(f"| {r.method} | {r.ensemble_size} | {r.num_bo_steps} | {r.total_evaluations} | {r.wall_clock_time:.2f} | {r.ensemble_mean_loss_default:.4f} | {r.mean_tailored_loss:.4f} | {r.anomaly_count} |\n")
        f.write("\n")

    print(f"Artifacts successfully saved to {out_dir}")


if __name__ == "__main__":
    main()
