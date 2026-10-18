import time
from typing import Optional
import numpy as np

try:
    from .common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS
except (ImportError, ValueError):
    try:
        from payoffnash.common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS
    except ImportError:
        from common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS


class NESeeking(NEEstimator):
    """
    Implements the Nash Equilibrium Seeking algorithm for noncooperative games 
    using extremum seeking with projection, as described by Frihauf et al.
    """

    def __init__(
        self,
        game: Game,
        k: np.ndarray,
        a: np.ndarray,
        omega: np.ndarray,
        omega_h: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ):
        self.game = game
        self.m = len(game.action_sets[0])
        self.dt = dt

        self.k = np.asarray(k, dtype=float).reshape(self.game.N, self.m)
        self.a = np.asarray(a, dtype=float).reshape(self.game.N, self.m)
        self.omega = np.asarray(omega, dtype=float).reshape(self.game.N, self.m)
        self.omega_h = (
            np.asarray(omega_h, dtype=float).reshape(self.game.N)
            if omega_h is not None
            else np.full(self.game.N, 10.0, dtype=float)
        )

        self.lb = np.array([[iv[0] for iv in s] for s in game.action_sets], dtype=float)
        self.ub = np.array([[iv[1] for iv in s] for s in game.action_sets], dtype=float)

    @classmethod
    def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
        m = len(game.action_sets[0])
        return {
            "k": (game.N, m),
            "a": (game.N, m),
            "omega": (game.N, m),
            "omega_h": (game.N,),
            "dt": (1,),
        }

    def estimate(
        self,
        initial_u_hat: np.ndarray,
        max_rel_error: float = 0.0,
        num_steps: int = MAX_STEPS,
        time_limit: float = np.inf,
        **kwargs,
    ) -> EstimateHistory:
        if (max_rel_error == 0.0) and num_steps == MAX_STEPS and time_limit == np.inf:
            raise ValueError("At least one stopping criterion must be specified: max_rel_error, num_steps, or time_limit.")

        track_time = (time_limit != np.inf)
        track_error = (max_rel_error > 0.0)

        history_length = 10000 if num_steps == MAX_STEPS else num_steps + 1
        total_dim = self.game.N * self.m

        u_hat_history = np.empty((history_length, total_dim), dtype=float)
        action_history = np.empty((history_length, total_dim), dtype=float)
        payoff_history = np.empty((history_length, self.game.N), dtype=float)

        u_hat = np.asarray(initial_u_hat, dtype=float).reshape(self.game.N, self.m).copy()

        u_0 = np.clip(u_hat.copy(), self.lb, self.ub)
        flat_u0 = u_0.ravel()
        eta = np.array([self.game.payoff_functions[i](*flat_u0) for i in range(self.game.N)], dtype=float)

        # Row 0 stores initial conditions at t = 0
        u_hat_history[0, :] = u_hat.ravel()
        action_history[0, :] = flat_u0
        payoff_history[0, :] = eta

        # Windowed averaging setup for max_rel_error (5 dither periods, non-overlapping)
        if track_error:
            omega_min = float(np.min(self.omega))
            n_period = max(1, int(round((2.0 * np.pi) / (omega_min * self.dt))))
            window_size = 5 * n_period
            current_window_sum = np.zeros(total_dim, dtype=float)
            current_window_steps = 0
            prev_window_avg = None
            rel_error = np.inf

        step = 0
        start_time = time.time() if track_time else 0.0
        elapsed_time = 0.0

        while True:
            if step >= num_steps:
                break
            if track_time and elapsed_time >= time_limit:
                break
            if track_error and rel_error <= max_rel_error:
                break

            step += 1
            if step >= history_length:
                new_length = history_length * 2
                new_u_hat_history = np.empty((new_length, total_dim), dtype=float)
                new_u_hat_history[:history_length, :] = u_hat_history
                u_hat_history = new_u_hat_history

                new_action_history = np.empty((new_length, total_dim), dtype=float)
                new_action_history[:history_length, :] = action_history
                action_history = new_action_history

                new_payoff_history = np.empty((new_length, self.game.N), dtype=float)
                new_payoff_history[:history_length, :] = payoff_history
                payoff_history = new_payoff_history

                history_length = new_length

            t = step * self.dt
            dist = np.minimum(u_hat - self.lb, self.ub - u_hat)
            alpha_i = np.clip(dist / self.a, 0.0, 1.0)
            mu = alpha_i * self.a * np.sin(self.omega * t)
            u = np.clip(u_hat + mu, self.lb, self.ub)

            flat_u = u.ravel()
            action_history[step, :] = flat_u

            payoffs = np.array([self.game.payoff_functions[i](*flat_u) for i in range(self.game.N)], dtype=float)
            payoff_history[step, :] = payoffs

            payoff_hpf = payoffs - eta
            eta += self.dt * self.omega_h * payoff_hpf

            phi_val = mu * payoff_hpf[:, None]
            proj_val = np.where((u_hat <= self.lb) & (phi_val < 0.0), 0.0, phi_val)
            proj_val = np.where((u_hat >= self.ub) & (proj_val > 0.0), 0.0, proj_val)

            u_hat = np.clip(u_hat + self.dt * self.k * proj_val, self.lb, self.ub)
            flat_u_hat = u_hat.ravel()
            u_hat_history[step, :] = flat_u_hat

            if track_error:
                current_window_sum += flat_u_hat
                current_window_steps += 1
                if current_window_steps == window_size:
                    current_window_avg = current_window_sum / window_size
                    if prev_window_avg is not None:
                        diff = np.linalg.norm(current_window_avg - prev_window_avg)
                        base = np.linalg.norm(prev_window_avg)
                        rel_error = float(diff / (base + 1e-12))
                    prev_window_avg = current_window_avg.copy()
                    current_window_sum = np.zeros(total_dim, dtype=float)
                    current_window_steps = 0

            if track_time and (step % 100 == 0):
                elapsed_time = time.time() - start_time

        return EstimateHistory(
            state_history=u_hat_history[:step + 1],
            action_history=action_history[:step + 1],
            payoff_history=payoff_history[:step + 1]
        )


class NESeeking5D(NESeeking):
    def __init__(
        self,
        game: Game,
        k: np.ndarray,
        a: np.ndarray,
        omega1: float | np.ndarray,
        dt: float = 0.001,
        omega2_ratio: float = 1.35,
        cutoff_ratio: float = 0.20,
    ):
        w1 = float(np.asarray(omega1).ravel()[0])
        w2 = omega2_ratio * w1
        omega = np.array([[w1], [w2]], dtype=float)
        omega_h = np.array([cutoff_ratio * w1, cutoff_ratio * w2], dtype=float)
        super().__init__(game=game, k=k, a=a, omega=omega, omega_h=omega_h, dt=dt)

    @classmethod
    def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
        m = len(game.action_sets[0])
        return {
            "k": (game.N, m),
            "a": (game.N, m),
            "omega1": (1,),
            "dt": (1,),
        }


if __name__ == "__main__":
    N = 2
    action_sets = [[(0.0, 300.0)], [(0.0, 300.0)]]
    Sd, p, m1, m2 = 100.0, 0.2, 30.0, 30.0

    def J1(u1, u2):
        return (Sd - (u1 - u2) / p) * (u1 - m1)

    def J2(u1, u2):
        return ((u1 - u2) / p) * (u2 - m2)

    game = Game(N=N, action_sets=action_sets, payoff_functions=[J1, J2])

    a = np.array([[0.075], [0.05]])
    k = np.array([[2.0], [5.0]])
    omega = np.array([[26.75], [22.0]])
    omega_h = np.array([10.0, 10.0])
    dt = 0.001
    num_steps = 1000000
    max_rel_error = 1e-7

    estimator = NESeeking(game=game, k=k, a=a, omega=omega, omega_h=omega_h, dt=dt)
    initial_actions = np.array([[50.0], [36.6667]])

    history = estimator.estimate(initial_actions, num_steps=num_steps, max_rel_error=max_rel_error)

    print(f"\nEstimated Nash Equilibrium (u_hat):")
    print(f"Firm 1 Price (u1): {history.state_history[-1][0]}")
    print(f"Firm 2 Price (u2): {history.state_history[-1][1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")