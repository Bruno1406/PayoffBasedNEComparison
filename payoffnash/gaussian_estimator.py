import inspect
from typing import Optional
import numpy as np
from collections.abc import Callable
# from src.payoffnash.common_types import ActionSet, PayoffFunction

PayoffFunction = Callable[..., float]
ActionSet = list[tuple[float, float]]

class GaussianEstimator:

    def __init__(self, N: int, action_sets: list[ActionSet], payoff_functions: list[PayoffFunction], v: float = 3, a: float = 1.0, epsilon: float = 1e-6):
        self.N = N
        self.action_sets = action_sets
        self.payoff_functions = payoff_functions
        self.v = v
        self.a = a
        self.epsilon = epsilon
        
        if len(action_sets) != N or len(payoff_functions) != N:
            raise ValueError("Length of action_sets and payoff_functions must be equal to N.")
        
        action_set_dim = sum([len(action_set) for action_set in action_sets])
        for payoff_function in payoff_functions:
            if not callable(payoff_function):
                raise ValueError("All elements in payoff_functions must be callable.")
            sig = inspect.signature(payoff_function)
            if len(sig.parameters) != action_set_dim:
                raise ValueError(f"Each payoff function must accept exactly {action_set_dim} parameters.")
        
        # Optimization 2: Precompute midpoints and sizes for _contrain_set
        self.action_set_midpoints = []
        self.action_set_sizes = []
        for action_set in action_sets:
            midpoints = [(interval[0] + interval[1]) / 2 for interval in action_set]
            sizes = [interval[1] - interval[0] for interval in action_set]
            self.action_set_midpoints.append(midpoints)
            self.action_set_sizes.append(sizes)
            
    def _contrain_set(self, player_idx: int, rho: float) -> ActionSet:
        constrained_set = []
        # Fetch precomputed values
        midpoints = self.action_set_midpoints[player_idx]
        sizes = self.action_set_sizes[player_idx]
        
        for i in range(len(midpoints)):
            delta = (sizes[i] / 2) * rho
            constrained_set.append((midpoints[i] - delta, midpoints[i] + delta))
        return constrained_set
    
    def _project_to_action_set(self, action: np.ndarray, action_set: ActionSet) -> np.ndarray:
        projected_action = np.copy(action)
        for i, interval in enumerate(action_set):
            projected_action[i] = np.clip(projected_action[i], interval[0], interval[1])
        return projected_action
            
    def estimate(self, initial_actions: list[np.ndarray], max_rel_error: Optional[float] = None, N_iters: Optional[int] = None
                )  -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        state = [np.copy(act) for act in initial_actions]
        # ADDED: History tracking
        action_history = []
        payoff_history = []
        
        flat_nash_estimate = np.full(sum(len(s) for s in self.action_sets), np.inf, dtype=float)
        rel_error = np.inf
        t = 1
        
        continue_condition = lambda: rel_error > max_rel_error if max_rel_error is not None else t <= N_iters
        
        while continue_condition():
            sigma = self.a / np.power(t, 1/4)
            gama = 4 / (self.v * t)
            rho = 1 / (np.power(t, 1/4 - self.epsilon))
            
            samples, actions = [], []
            for i in range(self.N):
                sample = np.random.normal(state[i], sigma, size=len(self.action_sets[i]))
                samples.append(sample)
                actions.append(self._project_to_action_set(sample, self.action_sets[i]))
            
            flat_actions = np.concatenate(actions)
            # ADDED: Record history
            action_history.append(flat_actions.copy())
            
            payoffs = []
            for i in range(self.N):
                p = self.payoff_functions[i](*flat_actions)
                payoffs.append(p) 
                gradient_estimate = p * (samples[i] - state[i]) / (sigma**2)
                state[i] = self._project_to_action_set(state[i] + gama * gradient_estimate, self._contrain_set(i, 1 - rho))
            
            # ADDED: Record payoffs
            payoff_history.append(payoffs)
            
            flat_state = np.concatenate(state)
            rel_error = np.linalg.norm(flat_state - flat_nash_estimate)
            flat_nash_estimate = flat_state.copy()
            t += 1
            
        return state, action_history, payoff_history


if __name__ == "__main__":
    # --- Paper's Duopoly Price Game Example ---
    N = 2
    
    # The paper notes the action set is all positive real numbers.
    # We use a large finite upper bound to prevent np.inf errors in your _contrain_set method.
    action_sets = [[(0.0, 300.0)], [(0.0, 300.0)]]

    # Simulation parameters from the paper
    Sd = 100.0
    p = 0.2
    m1 = 30.0
    m2 = 30.0

    # Payoff Functions (Returning NEGATIVE profit for Gradient Descent)
    def J1(u1, u2):
        # Sales model for Firm 1
        s1 = Sd - (u1 - u2) / p
        profit = s1 * (u1 - m1)
        return profit 
    
    def J2(u1, u2):
        # Sales model for Firm 2
        s2 = (u1 - u2) / p
        profit = s2 * (u2 - m2)
        return profit
    
    payoff_functions = [J1, J2]

    # Initialize the estimator
    estimator = GaussianEstimator(N, action_sets, payoff_functions, v=3, a=100, epsilon=1e-6)
    
    # The paper initializes the simulation at u1 = 50 and u2 = 36.66
    initial_actions = [np.array([10.0]), np.array([30.0])]
    
    print("Starting estimation for the paper's duopoly price game...")
    nash_estimate = estimator.estimate(initial_actions, N_iters=10000)
    
    print(f"\nEstimated Nash Equilibrium:")
    print(f"Firm 1 Price (u1): {nash_estimate[0][0]}")
    print(f"Firm 2 Price (u2): {nash_estimate[0][1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")