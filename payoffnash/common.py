from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import inspect
import numpy as np

PayoffFunction = Callable[..., float]
ActionSet = list[tuple[float, float]]


MAX_STEPS: int = 0xFFFFFFFF


@dataclass
class EstimateHistory:
    state_history: np.ndarray
    action_history: np.ndarray
    payoff_history: np.ndarray

    def __post_init__(self):
        if self.state_history.ndim != 2:
            raise ValueError("state_history must be a 2D array.")
        if self.action_history.ndim != 2:
            raise ValueError("action_history must be a 2D array.")
        if self.payoff_history.ndim != 2:
            raise ValueError("payoff_history must be a 2D array.")

        if self.state_history.shape != self.action_history.shape:
            raise ValueError("state_history and action_history must have the same shape.")
        if self.state_history.shape[0] != self.payoff_history.shape[0]:
            raise ValueError("state_history and payoff_history must have the same number of rows.")

@dataclass
class Game:
    N: int
    action_sets: list[ActionSet]
    payoff_functions: list[PayoffFunction]

    def __post_init__(self):
        if len(self.action_sets) != self.N or len(self.payoff_functions) != self.N:
            raise ValueError("Length of action_sets and payoff_functions must be equal to N.")
        
        action_set_dim = sum([len(action_set) for action_set in self.action_sets])
        for payoff_function in self.payoff_functions:
            if not callable(payoff_function):
                raise ValueError("All elements in payoff_functions must be callable.")
            sig = inspect.signature(payoff_function)
            if len(sig.parameters) != action_set_dim:
                raise ValueError(f"Each payoff function must accept exactly {action_set_dim} parameters.")


class NEEstimator(ABC):
    game: Game

    @abstractmethod
    def estimate(
        self, initial_actions: np.ndarray, **kwargs
    ) -> EstimateHistory:
        pass

    @classmethod
    @abstractmethod
    def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
        pass


def quadratic_game_regret_cost(game: Any, history: EstimateHistory) -> float:
    actions = history.action_history
    H = game.H
    d = game.d
    lb = np.array([[iv[0] for iv in s] for s in game.action_sets]).ravel()
    ub = np.array([[iv[1] for iv in s] for s in game.action_sets]).ravel()

    u0 = actions[:, 0]
    u1 = actions[:, 1]

    u0_unc = -(H[0, 1] * u1 + d[0]) / H[0, 0]
    u0_br = np.clip(u0_unc, lb[0], ub[0])

    u1_unc = -(H[1, 0] * u0 + d[1]) / H[1, 1]
    u1_br = np.clip(u1_unc, lb[1], ub[1])

    diff0 = (
        0.5 * H[0, 0] * (u0_br**2 - u0**2)
        + H[0, 1] * (u0_br - u0) * u1
        + d[0] * (u0_br - u0)
    )
    regret0 = np.maximum(diff0, 0.0)

    diff1 = (
        0.5 * H[1, 1] * (u1_br**2 - u1**2)
        + H[1, 0] * u0 * (u1_br - u1)
        + d[1] * (u1_br - u1)
    )
    regret1 = np.maximum(diff1, 0.0)

    total_regret = float(np.sum(regret0 + regret1))
    return float(np.log10(max(total_regret, 1e-12)))