import inspect
from typing import Optional, Callable
import numpy as np

PayoffFunction = Callable[..., float]
ActionSet = list[tuple[float, float]]

class NE_Seeking:
    """
    Implements the Nash Equilibrium Seeking algorithm for noncooperative games 
    using extremum seeking with projection, as described by Frihauf et al.
    """

    def __init__(self, N: int, action_sets: list[ActionSet], payoff_functions: list[PayoffFunction],
                 k: list[np.ndarray], a: list[np.ndarray], omega: list[np.ndarray], 
                 phi: list[np.ndarray], omega_h: list[float] = None, dt: float = 0.01):
        self.N = N
        self.action_sets = action_sets
        self.payoff_functions = payoff_functions
        
        # Extremum seeking parameters
        self.k = k          # Integration gains
        self.a = a          # Perturbation amplitudes
        self.omega = omega  # Perturbation frequencies
        self.phi = phi      # Phase shifts
        self.dt = dt        # Forward Euler time step
        
        # Washout filter cutoff frequencies (defaults to 10.0 if not provided)
        self.omega_h = omega_h if omega_h is not None else [10.0] * N

        if len(action_sets) != N or len(payoff_functions) != N:
            raise ValueError("Length of action_sets and payoff_functions must be equal to N.")
        
        action_set_dim = sum([len(action_set) for action_set in action_sets])
        for payoff_function in payoff_functions:
            if not callable(payoff_function):
                raise ValueError("All elements in payoff_functions must be callable.")
            sig = inspect.signature(payoff_function)
            if len(sig.parameters) != action_set_dim:
                raise ValueError(f"Each payoff function must accept exactly {action_set_dim} parameters.")

    def _project(self, phi_val: np.ndarray, u_hat: np.ndarray, player_idx: int) -> np.ndarray:
        """
        Implements the Projection operator from Eq (81) of Frihauf et al.,
        extended to handle BOTH lower and upper bounds of the action set.
        """
        proj_val = np.zeros_like(phi_val)
        for d in range(len(phi_val)):
            lower_bound = self.action_sets[player_idx][d][0]
            upper_bound = self.action_sets[player_idx][d][1]
            
            # If at or below lower bound and trying to go lower
            if u_hat[d] <= lower_bound and phi_val[d] < 0:
                proj_val[d] = 0.0
            # If at or above upper bound and trying to go higher
            elif u_hat[d] >= upper_bound and phi_val[d] > 0:
                proj_val[d] = 0.0
            else:
                proj_val[d] = phi_val[d]
        return proj_val

    def estimate(self, initial_u_hat: list[np.ndarray], num_steps: int
                ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        u_hat = [np.copy(u) for u in initial_u_hat]
        action_history = []
        payoff_history = []
        
        # Initialize LPF states (eta) for the washout filter
        # We initialize them to the initial payoffs to prevent a massive kick on step 1
        initial_u = [u_hat[i] + self.a[i] * np.sin(self.phi[i]) for i in range(self.N)]
        flat_initial_u = np.concatenate(initial_u)
        eta = [np.full_like(u_hat[i], self.payoff_functions[i](*flat_initial_u)) for i in range(self.N)]
        
        
        for step in range(num_steps):
            t = step * self.dt
            mu = [self.a[i] * np.sin(self.omega[i] * t + self.phi[i]) for i in range(self.N)]
            u = [u_hat[i] + mu[i] for i in range(self.N)]
            
            # Record actions (u)
            action_history.append(np.concatenate(u).copy())
            
            payoffs = []
            for i in range(self.N):
                payoffs.append(self.payoff_functions[i](*np.concatenate(u)))
            
            # Record payoffs
            payoff_history.append(payoffs)
            
            for i in range(self.N):
                eta[i] = eta[i] + self.dt * self.omega_h[i] * (payoffs[i] - eta[i])
                payoff_hpf = payoffs[i] - eta[i]
                phi_val = mu[i] * payoff_hpf
                proj_val = self._project(phi_val, u_hat[i], i)
                u_hat[i] = u_hat[i] + self.dt * self.k[i] * proj_val

        return u_hat, action_history, payoff_history

if __name__ == "__main__":
    # --- Paper's Duopoly Price Game Example ---
    N = 2
    
    action_sets = [[(0.0, 300.0)], [(0.0, 300.0)]]

    # Simulation parameters
    Sd = 100.0
    p = 0.2
    m1 = 30.0
    m2 = 30.0

    # Payoff Functions (Returning POSITIVE profit, as Extremum Seeking maximizes)
    def J1(u1, u2):
        s1 = Sd - (u1 - u2) / p
        profit = s1 * (u1 - m1)
        return profit 
    
    def J2(u1, u2):
        s2 = (u1 - u2) / p
        profit = s2 * (u2 - m2)
        return profit
    
    payoff_functions = [J1, J2]

    # Extremum seeking parameters 
    a = [np.array([0.075]), np.array([0.05])]
    k = [np.array([2.0]), np.array([5.0])]
    omega = [np.array([26.75]), np.array([22.0])]
    phi = [np.array([0.0]), np.array([0.0])]
    
    # Washout filter cutoff frequencies
    omega_h = [10.0, 10.0]

    # Discretization delta (dt)
    dt = 0.001 
    num_steps = 100000  # corresponds to 250 seconds from the paper

    # Initialize the estimator
    estimator = NE_Seeking(
        N=N, 
        action_sets=action_sets, 
        payoff_functions=payoff_functions, 
        k=k, a=a, omega=omega, phi=phi, omega_h=omega_h, dt=dt
    )
    
    initial_actions = [np.array([50.0]), np.array([36.6667])]  # Initial guesses for u_hat
    
    print("Starting forward euler estimation for the Frihauf Duopoly Price Game...")
    nash_estimate = estimator.estimate(initial_actions, num_steps=num_steps)
    
    print(f"\nEstimated Nash Equilibrium (u_hat):")
    print(f"Firm 1 Price (u1): {nash_estimate[0][0]}")
    print(f"Firm 2 Price (u2): {nash_estimate[0][1]}")
    print("(Expected theoretical exact result: 43.3333, 36.6666)")