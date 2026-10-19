from typing import Optional
import numpy as np
import time

try:
    from .common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS
except (ImportError, ValueError):
    try:
        from payoffnash.common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS
    except ImportError:
        from common import ActionSet, Game, NEEstimator, PayoffFunction, EstimateHistory, MAX_STEPS


class ZOGrandientPlay(NEEstimator):
    """
    Implements the Payoff-Based Learning algorithm for Nash Equilibrium seeking
    in strongly monotone games using Gaussian perturbations and regularized set projections,
    based on Tatarenko & Kamgarpour (2022) / (IEEE TAC 2019).

    Note on Optimization Direction:
    While the original paper formulates the problem for cost minimization using
    gradient descent (mu <- mu - gamma * m), this class implements GRADIENT ASCENT
    (mu <- mu + gamma * m) to maximize payoff/profit functions directly, ensuring
    full compatibility with payoff-maximization frameworks like NESeeking.
    """

    def __init__(
        self,
        game: Game,
        v: float = 3.0,
        a: float = 1.0,
        epsilon: float = 1e-6,
    ):
        self.game = game
        self.v = v
        self.a = a
        self.epsilon = epsilon

        self.m = len(game.action_sets[0])
        self.total_dim = game.N * self.m

        self.lb = np.array([[iv[0] for iv in s] for s in game.action_sets], dtype=float)
        self.ub = np.array([[iv[1] for iv in s] for s in game.action_sets], dtype=float)
        self.midpoints = (self.lb + self.ub) / 2.0
        self.sizes = self.ub - self.lb

    @classmethod
    def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
        return {"v": (1,), "a": (1,), "epsilon": (1,)}

    def estimate(
        self,
        initial_actions: np.ndarray,
        max_rel_error: float = 0.0,
        num_steps: int = MAX_STEPS,
        time_limit: float = np.inf,
        true_ne: Optional[np.ndarray] = None,
        ne_tol: float = 0.0,
        **kwargs,
    ) -> EstimateHistory:
        if (
            (max_rel_error == 0.0)
            and num_steps == MAX_STEPS
            and time_limit == np.inf
            and (true_ne is None or ne_tol <= 0.0)
        ):
            raise ValueError(
                "At least one stopping criterion must be specified: max_rel_error, num_steps, time_limit, or (true_ne, ne_tol)."
            )

        track_time = (time_limit != np.inf)
        track_error = (max_rel_error > 0.0)
        track_ne = (true_ne is not None and ne_tol > 0.0)
        flat_true_ne = np.asarray(true_ne, dtype=float).ravel() if track_ne else None

        history_length = 10000 if num_steps == MAX_STEPS else num_steps + 1
        total_dim = self.game.N * self.m

        state_history = np.empty((history_length, total_dim), dtype=float)
        action_history = np.empty((history_length, total_dim), dtype=float)
        payoff_history = np.empty((history_length, self.game.N), dtype=float)

        state = np.asarray(initial_actions, dtype=float).reshape(self.game.N, self.m).copy()
        flat_state0 = state.ravel()

        state_history[0, :] = flat_state0
        action_history[0, :] = flat_state0
        payoff_history[0, :] = np.array([self.game.payoff_functions[i](*flat_state0) for i in range(self.game.N)], dtype=float)

        flat_prev_state = flat_state0.copy() if track_error else None
        rel_error = np.inf
        start_time = time.time() if track_time else 0.0
        elapsed_time = 0.0
        t = 1

        while True:
            if t > num_steps:
                break
            if track_time and elapsed_time >= time_limit:
                break
            if track_error and rel_error <= max_rel_error:
                break

            if t >= history_length:
                new_length = history_length * 2
                new_action_history = np.empty((new_length, total_dim), dtype=float)
                new_action_history[:history_length, :] = action_history
                action_history = new_action_history

                new_payoff_history = np.empty((new_length, self.game.N), dtype=float)
                new_payoff_history[:history_length, :] = payoff_history
                payoff_history = new_payoff_history

                new_state_history = np.empty((new_length, total_dim), dtype=float)
                new_state_history[:history_length, :] = state_history
                state_history = new_state_history

                history_length = new_length

            sigma = self.a / np.power(t, 0.25)
            gama = 4.0 / (self.v * t)
            rho = 1.0 / np.power(t, 0.25 - self.epsilon)
            shrink_factor = float(np.clip(1.0 - rho, 1e-4, 1.0))

            samples = np.random.normal(state, sigma)
            actions = np.clip(samples, self.lb, self.ub)
            flat_actions = actions.ravel()
            action_history[t, :] = flat_actions

            payoffs = np.array([self.game.payoff_functions[i](*flat_actions) for i in range(self.game.N)], dtype=float)
            payoff_history[t, :] = payoffs

            grad_est = (payoffs[:, None] * (samples - state)) / (sigma**2)

            delta = (self.sizes / 2.0) * shrink_factor
            reg_lb = self.midpoints - delta
            reg_ub = self.midpoints + delta
            state = np.clip(state + gama * grad_est, reg_lb, reg_ub)
            flat_state = state.ravel()
            state_history[t, :] = flat_state

            if track_ne:
                dist_ne = float(np.linalg.norm(flat_state - flat_true_ne))
                if dist_ne <= ne_tol:
                    t += 1
                    break

            if track_error:
                diff_norm = np.linalg.norm(flat_state - flat_prev_state)
                prev_norm = np.linalg.norm(flat_prev_state)
                rel_error = float(diff_norm if np.isinf(prev_norm) else diff_norm / (prev_norm + 1e-12))
                flat_prev_state = flat_state.copy()

            if track_time and (t % 100 == 0):
                elapsed_time = time.time() - start_time

            t += 1

        return EstimateHistory(
            state_history=state_history[:t],
            action_history=action_history[:t],
            payoff_history=payoff_history[:t]
        )


if __name__ == "__main__":
    N = 2
    action_sets = [[(0.0, 300.0)], [(0.0, 300.0)]]

    Sd = 100.0
    p = 0.2
    m1 = 30.0
    m2 = 30.0

    def J1(u1, u2):
        s1 = Sd - (u1 - u2) / p
        return s1 * (u1 - m1)

    def J2(u1, u2):
        s2 = (u1 - u2) / p
        return s2 * (u2 - m2)

    game = Game(N, action_sets, [J1, J2])
    estimator = ZOGrandientPlay(game, v=3, a=100, epsilon=1e-6)
    initial_actions = np.array([[10.0], [30.0]])

    print("Starting estimation for the paper's duopoly price game (one-point)...")
    history = estimator.estimate(initial_actions, num_steps=100000, max_rel_error=1e-7)

    print(f"\nEstimated Nash Equilibrium (one-point):")
    print(f"Firm 1 Price (u1): {history.state_history[-1][0]}")
    print(f"Firm 2 Price (u2): {history.state_history[-1][1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")