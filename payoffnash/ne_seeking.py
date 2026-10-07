from typing import Optional
import numpy as np

try:
    from .common import ActionSet, Game, NEEstimator, PayoffFunction
except (ImportError, ValueError):
    try:
        from payoffnash.common import ActionSet, Game, NEEstimator, PayoffFunction
    except ImportError:
        from common import ActionSet, Game, NEEstimator, PayoffFunction


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
        phi: np.ndarray,
        omega_h: Optional[np.ndarray] = None,
        dt: float = 0.01,
    ):
        self.game = game
        self.m = len(game.action_sets[0])
        self.dt = dt

        self.k = np.asarray(k, dtype=float).reshape(self.game.N, self.m)
        self.a = np.asarray(a, dtype=float).reshape(self.game.N, self.m)
        self.omega = np.asarray(omega, dtype=float).reshape(self.game.N, self.m)
        self.phi = np.asarray(phi, dtype=float).reshape(self.game.N, self.m)
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
            "phi": (game.N, m),
            "dt": (1,),
        }

    def estimate(
        self,
        initial_u_hat: np.ndarray,
        num_steps: Optional[int] = None,
        num_iters: Optional[int] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        steps = num_steps if num_steps is not None else num_iters
        if steps is None:
            raise ValueError("num_steps or num_iters must be provided.")

        u_hat = np.asarray(initial_u_hat, dtype=float).reshape(self.game.N, self.m).copy()
        action_history = np.empty((steps, self.game.N * self.m), dtype=float)
        payoff_history = np.empty((steps, self.game.N), dtype=float)

        dist_0 = np.minimum(u_hat - self.lb, self.ub - u_hat)
        alpha_0 = np.clip(dist_0 / self.a, 0.0, 1.0)
        u_0 = np.clip(u_hat + alpha_0 * self.a * np.sin(self.phi), self.lb, self.ub)
        flat_u0 = u_0.ravel()
        eta = np.array([self.game.payoff_functions[i](*flat_u0) for i in range(self.game.N)], dtype=float)

        for step in range(steps):
            t = step * self.dt
            dist = np.minimum(u_hat - self.lb, self.ub - u_hat)
            alpha_i = np.clip(dist / self.a, 0.0, 1.0)
            mu = alpha_i * self.a * np.sin(self.omega * t + self.phi)
            u = np.clip(u_hat + mu, self.lb, self.ub)

            flat_u = u.ravel()
            action_history[step] = flat_u

            payoffs = np.array([self.game.payoff_functions[i](*flat_u) for i in range(self.game.N)], dtype=float)
            payoff_history[step] = payoffs

            payoff_hpf = payoffs - eta
            eta += self.dt * self.omega_h * payoff_hpf

            phi_val = mu * payoff_hpf[:, None]
            proj_val = np.where((u_hat <= self.lb) & (phi_val < 0.0), 0.0, phi_val)
            proj_val = np.where((u_hat >= self.ub) & (proj_val > 0.0), 0.0, proj_val)

            u_hat = np.clip(u_hat + self.dt * self.k * proj_val, self.lb, self.ub)

        return u_hat, action_history, payoff_history


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
    phi = np.array([[0.0], [0.0]])
    omega_h = np.array([10.0, 10.0])
    dt = 0.001
    num_steps = 100000

    estimator = NESeeking(game=game, k=k, a=a, omega=omega, phi=phi, omega_h=omega_h, dt=dt)
    initial_actions = np.array([[50.0], [36.6667]])

    print("Starting forward Euler estimation for the Frihauf Duopoly Price Game...")
    nash_estimate, _, _ = estimator.estimate(initial_actions, num_steps=num_steps)

    print(f"\nEstimated Nash Equilibrium (u_hat):")
    print(f"Firm 1 Price (u1): {nash_estimate[0]}")
    print(f"Firm 2 Price (u2): {nash_estimate[1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")