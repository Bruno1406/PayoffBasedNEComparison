from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import inspect
import numpy as np

PayoffFunction = Callable[..., float]
ActionSet = list[tuple[float, float]]

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @classmethod
    @abstractmethod
    def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
        pass