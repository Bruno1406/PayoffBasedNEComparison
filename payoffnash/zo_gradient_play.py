from typing import Optional
import numpy as np

try:
    from .common import ActionSet, Game, NEEstimator, PayoffFunction
except (ImportError, ValueError):
    try:
        from payoffnash.common import ActionSet, Game, NEEstimator, PayoffFunction
    except ImportError:
        from common import ActionSet, Game, NEEstimator, PayoffFunction


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
        estimation_type: str = "one-point",
    ):
        self.game = game
        self.v = v
        self.a = a
        self.epsilon = epsilon
        if estimation_type not in ("one-point", "two-point"):
            raise ValueError("estimation_type must be either 'one-point' or 'two-point'.")
        self.estimation_type = estimation_type

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
        max_rel_error: Optional[float] = None,
        N_iters: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_iters: Optional[int] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        iters = N_iters if N_iters is not None else (num_iters if num_iters is not None else num_steps)
        state = np.asarray(initial_actions, dtype=float).reshape(self.game.N, self.m).copy()
        action_history_list = []
        payoff_history_list = []

        flat_nash_estimate = np.full(self.total_dim, np.inf, dtype=float)
        rel_error = np.inf
        t = 1

        continue_condition = (
            lambda: rel_error > max_rel_error if max_rel_error is not None else t <= iters
        )

        while continue_condition():
            sigma = self.a / np.power(t, 0.25)
            gama = 4.0 / (self.v * t)
            rho = 1.0 / np.power(t, 0.25 - self.epsilon)
            shrink_factor = float(np.clip(1.0 - rho, 1e-4, 1.0))

            samples = np.random.normal(state, sigma)
            actions = np.clip(samples, self.lb, self.ub)
            flat_actions = actions.ravel()
            action_history_list.append(flat_actions)

            payoffs = np.array([self.game.payoff_functions[i](*flat_actions) for i in range(self.game.N)], dtype=float)
            payoff_history_list.append(payoffs)

            if self.estimation_type == "two-point":
                flat_state = state.ravel()
                p_mu = np.array([self.game.payoff_functions[i](*flat_state) for i in range(self.game.N)], dtype=float)
                grad_est = ((payoffs - p_mu)[:, None] * (samples - state)) / (sigma**2)
            else:
                grad_est = (payoffs[:, None] * (samples - state)) / (sigma**2)

            delta = (self.sizes / 2.0) * shrink_factor
            reg_lb = self.midpoints - delta
            reg_ub = self.midpoints + delta
            state = np.clip(state + gama * grad_est, reg_lb, reg_ub)

            flat_state = state.ravel()
            diff_norm = np.linalg.norm(flat_state - flat_nash_estimate)
            prev_norm = np.linalg.norm(flat_nash_estimate)
            rel_error = diff_norm if np.isinf(prev_norm) else diff_norm / (prev_norm + 1e-12)
            flat_nash_estimate = flat_state.copy()
            t += 1

        return state, np.array(action_history_list), np.array(payoff_history_list)


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
    nash_estimate, _, _ = estimator.estimate(initial_actions, N_iters=10000)

    print(f"\nEstimated Nash Equilibrium (one-point):")
    print(f"Firm 1 Price (u1): {nash_estimate[0]}")
    print(f"Firm 2 Price (u2): {nash_estimate[1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")

    estimator_2p = ZOGrandientPlay(game, v=3, a=100, epsilon=1e-6, estimation_type="two-point")
    print("\nStarting estimation for the paper's duopoly price game (two-point)...")
    nash_estimate_2p, _, _ = estimator_2p.estimate(initial_actions, N_iters=10000)
    print(f"\nEstimated Nash Equilibrium (two-point):")
    print(f"Firm 1 Price (u1): {nash_estimate_2p[0]}")
    print(f"Firm 2 Price (u2): {nash_estimate_2p[1]}")