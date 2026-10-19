from concurrent.futures import ProcessPoolExecutor
from typing import Optional
import numpy as np
from scipy.stats import qmc

try:
    from .common import Game, quadratic_game_regret_cost
    from .zo_gradient_play import ZOGrandientPlay
    from .ne_seeking import NESeeking5D
except (ImportError, ValueError):
    try:
        from payoffnash.common import Game, quadratic_game_regret_cost
        from payoffnash.zo_gradient_play import ZOGrandientPlay
        from payoffnash.ne_seeking import NESeeking5D
    except ImportError:
        from common import Game, quadratic_game_regret_cost
        from zo_gradient_play import ZOGrandientPlay
        from ne_seeking import NESeeking5D


def eval_zo_sobol_worker(args: tuple[Game, np.ndarray, int]) -> float:
    game, u_vec, t_sim = args
    bounds = np.array([[iv[1] - iv[0] for iv in s] for s in game.action_sets], dtype=float)
    min_w = float(np.min(bounds))
    a_val = float(10.0 ** (-2.301 + u_vec[0] * (0.301 - (-2.301)))) * min_w
    v_val = float(10.0 ** (-2.0 + u_vec[1] * (2.0 - (-2.0))))
    eps_val = float(10.0 ** (-6.0 + u_vec[2] * (-4.0 - (-6.0))))  # range [1e-6, 1e-4]
    est = ZOGrandientPlay(game, v=v_val, a=a_val, epsilon=eps_val)
    hist = est.estimate(np.zeros((game.N, 1)), num_steps=t_sim)
    return quadratic_game_regret_cost(game, hist)


def eval_ne_sobol_worker(args: tuple[Game, np.ndarray, int]) -> float:
    game, u_vec, t_sim = args
    bounds = np.array([[iv[1] - iv[0] for iv in s] for s in game.action_sets], dtype=float)
    w0, w1 = float(bounds[0, 0]), float(bounds[1, 0])
    k1 = float(10.0 ** (-2.0 + u_vec[0] * 4.0))
    k2 = float(10.0 ** (-2.0 + u_vec[1] * 4.0))
    a1 = float(10.0 ** (-2.301 + u_vec[2] * (0.301 - (-2.301)))) * w0
    a2 = float(10.0 ** (-2.301 + u_vec[3] * (0.301 - (-2.301)))) * w1
    w1_val = float(10.0 ** (0.0 + u_vec[4] * (1.699 - 0.0)))
    est = NESeeking5D(game, k=np.array([[k1], [k2]]), a=np.array([[a1], [a2]]), omega1=w1_val, dt=0.001)
    hist = est.estimate(np.zeros((game.N, 1)), num_steps=t_sim)
    return quadratic_game_regret_cost(game, hist)


def run_game_sobol(
    game: Game,
    dim: int,
    worker_fn,
    n_sobol: int = 256,
    t_sim: int = 5000,
    seed: int = 42,
    max_workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    sampler = qmc.Sobol(d=2 * dim, scramble=True, seed=seed)
    sample = sampler.random(n_sobol)
    A = sample[:, :dim]
    B = sample[:, dim:]

    tasks = [(game, row, t_sim) for row in A] + [(game, row, t_sim) for row in B]
    for i in range(dim):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        tasks.extend([(game, row, t_sim) for row in AB_i])

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker_fn, tasks, chunksize=32))

    y_A = np.array(results[:n_sobol])
    y_B = np.array(results[n_sobol : 2 * n_sobol])
    var_y = float(np.var(np.concatenate([y_A, y_B]), ddof=1))
    if var_y < 1e-12:
        return np.zeros(dim), np.zeros(dim)

    S = np.zeros(dim)
    ST = np.zeros(dim)
    offset = 2 * n_sobol
    for i in range(dim):
        y_AB_i = np.array(results[offset + i * n_sobol : offset + (i + 1) * n_sobol])
        ST[i] = float(np.mean((y_A - y_AB_i) ** 2) / (2.0 * var_y))
        S[i] = float(np.mean(y_B * (y_AB_i - y_A)) / var_y)
    return S, ST


def run_convergence_worker(
    args: tuple[int, Game, dict[str, float], dict[str, float], float, int]
) -> dict[str, int | float | bool]:
    gid, game, zo_default, ne_default, ne_tol, max_steps = args
    u_star = game.nash_equilibrium.ravel()
    bounds = np.array([[iv[1] - iv[0] for iv in s] for s in game.action_sets], dtype=float)
    min_w = float(np.min(bounds))
    w0, w1 = float(bounds[0, 0]), float(bounds[1, 0])
    mid_actions = np.array([[0.5 * (iv[0] + iv[1]) for iv in s] for s in game.action_sets], dtype=float)

    # 1. ZOGP Default Convergence Runs (5 Random Seeds: 14062003 + k)
    zo_costs: list[float] = []
    zo_log_steps_list: list[float] = []
    zo_steps_list: list[int] = []
    zo_conv_list: list[bool] = []

    for k in range(5):
        np.random.seed(14062003 + k)
        est_zo = ZOGrandientPlay(
            game,
            v=zo_default["v"],
            a=zo_default["a"] * min_w,
            epsilon=zo_default["epsilon"],
        )
        h_zo = est_zo.estimate(mid_actions, num_steps=max_steps, true_ne=u_star, ne_tol=ne_tol)
        s_k = len(h_zo.action_history) - 1
        c_k = quadratic_game_regret_cost(game, h_zo)
        conv_k = bool(np.linalg.norm(h_zo.state_history[-1] - u_star) <= ne_tol)

        zo_steps_list.append(s_k)
        zo_log_steps_list.append(float(np.log10(max(s_k, 1))))
        zo_costs.append(c_k)
        zo_conv_list.append(conv_k)

    exp_zo_log_regret = float(np.mean(zo_costs))
    exp_zo_log_steps = float(np.mean(zo_log_steps_list))
    zo_steps = float(10.0 ** exp_zo_log_steps)
    zo_converged = bool(np.mean(zo_conv_list) >= 0.5)
    zo_conv_rate = float(np.mean(zo_conv_list))

    # 2. NESeeking5D Default Convergence Run (Deterministic)
    est_ne = NESeeking5D(
        game,
        k=np.array([[ne_default["k0"]], [ne_default["k1"]]]),
        a=np.array([[ne_default["a0"] * w0], [ne_default["a1"] * w1]]),
        omega1=ne_default["omega1"],
        dt=ne_default["dt"],
    )
    h_ne = est_ne.estimate(mid_actions, num_steps=max_steps, true_ne=u_star, ne_tol=ne_tol)
    ne_steps = len(h_ne.action_history) - 1
    ne_cost = quadratic_game_regret_cost(game, h_ne)
    ne_converged = bool(np.linalg.norm(h_ne.state_history[-1] - u_star) <= ne_tol)
    ne_log_steps = float(np.log10(max(ne_steps, 1)))

    return {
        "game_id": gid,
        "ne_tol": float(ne_tol),
        "zo_steps": zo_steps,
        "zo_log_steps": exp_zo_log_steps,
        "zo_log_regret": exp_zo_log_regret,
        "zo_converged": zo_converged,
        "zo_conv_rate": zo_conv_rate,
        "ne_steps": ne_steps,
        "ne_log_steps": ne_log_steps,
        "ne_log_regret": ne_cost,
        "ne_converged": ne_converged,
    }

