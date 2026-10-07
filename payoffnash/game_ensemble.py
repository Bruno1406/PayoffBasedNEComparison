import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from scipy.stats.qmc import LatinHypercube
except ImportError:
    class LatinHypercube:
        def __init__(self, d: int, seed: Optional[int] = None):
            self.d = d
            self.rng = np.random.default_rng(seed)

        def random(self, n: int) -> np.ndarray:
            result = np.empty((n, self.d), dtype=float)
            for j in range(self.d):
                perm = self.rng.permutation(n)
                result[:, j] = (perm + self.rng.uniform(0.0, 1.0, size=n)) / n
            return result

try:
    from .common import ActionSet, Game, PayoffFunction
except (ImportError, ValueError):
    try:
        from payoffnash.common import ActionSet, Game, PayoffFunction
    except ImportError:
        from common import ActionSet, Game, PayoffFunction


@dataclass
class QuadraticGame(Game):
    H: np.ndarray
    d: np.ndarray
    c: np.ndarray
    nash_equilibrium: np.ndarray
    optimal_payoffs: list[float]


def _make_payoff_function(
    player_idx: int,
    H_i: np.ndarray,
    H_ii: np.ndarray,
    d_i: np.ndarray,
    c_i: float,
    total_dim: int,
    m: int,
) -> Callable[..., float]:
    start_idx = player_idx * m
    end_idx = start_idx + m
    params = [inspect.Parameter(f"u_{k}", inspect.Parameter.POSITIONAL_OR_KEYWORD) for k in range(total_dim)]

    def payoff(*args: float) -> float:
        u = np.asarray(args, dtype=float)
        u_i = u[start_idx:end_idx]
        return float(u_i @ (H_i @ u + d_i) - 0.5 * (u_i @ H_ii @ u_i) + c_i)

    payoff.__signature__ = inspect.Signature(params)
    return payoff


class QuadraticGameGenerator:
    def __init__(
        self,
        N: int,
        m: int = 1,
        coeff_interval: tuple[float, float] = (-1.0, 1.0),
        action_lower_interval: tuple[float, float] = (-1.0, 0.0),
        action_upper_interval: tuple[float, float] = (0.0, 1.0),
        alpha: float = 1e-3,
        enforce_interior: bool = True,
        seed: Optional[int] = None,
        max_batches: int = 100,
    ):
        if N < 2:
            raise ValueError("Number of players N must be >= 2.")
        if m < 1:
            raise ValueError("Action dimension m must be >= 1.")
        if alpha <= 0:
            raise ValueError("Monotonicity margin alpha must be > 0.")
        if action_lower_interval[0] >= action_lower_interval[1] or action_lower_interval[1] > 0.0:
            raise ValueError("action_lower_interval must satisfy lb_min < lb_max <= 0.0.")
        if action_upper_interval[0] < 0.0 or action_upper_interval[0] >= action_upper_interval[1]:
            raise ValueError("action_upper_interval must satisfy 0.0 <= ub_min < ub_max.")

        self.N = N
        self.m = m
        self.total_dim = N * m
        self.coeff_interval = coeff_interval
        self.action_lower_interval = action_lower_interval
        self.action_upper_interval = action_upper_interval
        self.alpha = alpha
        self.enforce_interior = enforce_interior
        self.seed = seed
        self.max_batches = max_batches

        self.d_total = (self.total_dim) ** 2 + self.total_dim + self.N + 2 * self.total_dim
        self.sampler = LatinHypercube(d=self.d_total, seed=self.seed)

    def _scale_samples(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = samples.shape[0]
        c_min, c_max = self.coeff_interval
        lb_min, lb_max = self.action_lower_interval
        ub_min, ub_max = self.action_upper_interval

        pos = 0
        h_len = self.total_dim**2
        h_raw = c_min + (c_max - c_min) * samples[:, pos : pos + h_len]
        H = h_raw.reshape(k, self.total_dim, self.total_dim)
        pos += h_len

        d = c_min + (c_max - c_min) * samples[:, pos : pos + self.total_dim]
        pos += self.total_dim

        c = c_min + (c_max - c_min) * samples[:, pos : pos + self.N]
        pos += self.N

        lb = lb_min + (lb_max - lb_min) * samples[:, pos : pos + self.total_dim]
        pos += self.total_dim

        ub = ub_min + (ub_max - ub_min) * samples[:, pos : pos + self.total_dim]

        if self.m > 1:
            for i in range(self.N):
                s, e = i * self.m, (i + 1) * self.m
                H_ii = H[:, s:e, s:e]
                H[:, s:e, s:e] = 0.5 * (H_ii + np.swapaxes(H_ii, -1, -2))

        return H, d, c, lb, ub

    def generate(self, ensemble_size: int) -> list[QuadraticGame]:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")

        valid_games: list[QuadraticGame] = []
        batch_size = max(ensemble_size * 2, 200)
        total_drawn = 0

        for _ in range(self.max_batches):
            samples = self.sampler.random(n=batch_size)
            total_drawn += batch_size
            H, d, c, lb, ub = self._scale_samples(samples)

            H_sym = 0.5 * (H + np.swapaxes(H, -1, -2))
            eigvals = np.linalg.eigvalsh(H_sym)
            mono_mask = eigvals[:, -1] <= -self.alpha

            if not np.any(mono_mask):
                batch_size = min(batch_size * 2, 10000)
                continue

            sub_indices = np.where(mono_mask)[0]
            H_sub = H[mono_mask]
            d_sub = d[mono_mask]
            lb_sub = lb[mono_mask]
            ub_sub = ub[mono_mask]

            u_stars = np.linalg.solve(H_sub, -d_sub[..., None])[..., 0]

            if self.enforce_interior:
                interior_mask = np.all((u_stars > lb_sub) & (u_stars < ub_sub), axis=-1)
                passing_sub_indices = sub_indices[interior_mask]
                u_stars_valid = u_stars[interior_mask]
            else:
                passing_sub_indices = sub_indices
                u_stars_valid = u_stars

            needed = ensemble_size - len(valid_games)
            selected = passing_sub_indices[:needed]
            selected_u_stars = u_stars_valid[:needed]

            for i, orig_idx in enumerate(selected):
                u_star = selected_u_stars[i]
                H_val = H[orig_idx]
                d_val = d[orig_idx]
                c_val = c[orig_idx]
                lb_val = lb[orig_idx]
                ub_val = ub[orig_idx]

                action_sets: list[ActionSet] = []
                payoff_functions: list[PayoffFunction] = []

                for p in range(self.N):
                    s, e = p * self.m, (p + 1) * self.m
                    p_intervals = [(float(lb_val[k]), float(ub_val[k])) for k in range(s, e)]
                    action_sets.append(p_intervals)

                    H_i = H_val[s:e, :]
                    H_ii = H_val[s:e, s:e]
                    d_i = d_val[s:e]
                    c_i = float(c_val[p])

                    fn = _make_payoff_function(p, H_i, H_ii, d_i, c_i, self.total_dim, self.m)
                    payoff_functions.append(fn)

                flat_u_star = tuple(float(x) for x in u_star)
                optimal_payoffs = [float(f(*flat_u_star)) for f in payoff_functions]

                game = QuadraticGame(
                    N=self.N,
                    action_sets=action_sets,
                    payoff_functions=payoff_functions,
                    H=H_val,
                    d=d_val,
                    c=c_val,
                    nash_equilibrium=u_star,
                    optimal_payoffs=optimal_payoffs,
                )
                valid_games.append(game)

            if len(valid_games) >= ensemble_size:
                break

            p_hat = max(len(valid_games) / total_drawn, 1e-4)
            rem = ensemble_size - len(valid_games)
            batch_size = int(np.clip(np.ceil((rem / p_hat) * 1.3), 100, 10000))

        if len(valid_games) < ensemble_size:
            raise RuntimeError(
                f"Generated only {len(valid_games)}/{ensemble_size} valid games after {total_drawn} draws. "
                "Consider increasing max_batches or decreasing alpha."
            )

        return valid_games[:ensemble_size]


def generate_quadratic_games(
    ensemble_size: int,
    N: int = 2,
    m: int = 1,
    coeff_interval: tuple[float, float] = (-1.0, 1.0),
    action_lower_interval: tuple[float, float] = (-1.0, 0.0),
    action_upper_interval: tuple[float, float] = (0.0, 1.0),
    alpha: float = 1e-3,
    enforce_interior: bool = True,
    seed: Optional[int] = None,
    max_batches: int = 100,
) -> list[QuadraticGame]:
    generator = QuadraticGameGenerator(
        N=N,
        m=m,
        coeff_interval=coeff_interval,
        action_lower_interval=action_lower_interval,
        action_upper_interval=action_upper_interval,
        alpha=alpha,
        enforce_interior=enforce_interior,
        seed=seed,
        max_batches=max_batches,
    )
    return generator.generate(ensemble_size)


if __name__ == "__main__":
    ensemble = generate_quadratic_games(ensemble_size=3, N=2, m=1, seed=42)
    print(f"Generated {len(ensemble)} strongly monotone quadratic games:")
    for i, g in enumerate(ensemble):
        print(f"\nGame {i + 1}:")
        print(f"  Action sets: {g.action_sets}")
        print(f"  Nash equilibrium u*: {g.nash_equilibrium}")
        print(f"  Optimal payoffs: {g.optimal_payoffs}")
        print(f"  Hessian eigenvalues: {np.linalg.eigvalsh(0.5 * (g.H + g.H.T))}")

