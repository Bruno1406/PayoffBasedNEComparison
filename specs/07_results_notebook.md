# Specification 07: Final Results Notebook & Convergence Architecture (`results.ipynb`)

## 1. Overview & Objectives
This specification defines the architecture, mathematical formulations, interfaces, validation criteria, and execution structure for `results.ipynb`—the primary synthesis notebook producing all final figures, tables, and comparative analyses for the thesis.

The notebook evaluates **NESeeking** and **ZOGrandientPlay** across:
1. **The 5 Representative Games** (Game 11: ill-conditioned; Game 70: isotropic; Game 97: narrow bounds; Game 40: asymmetric coupling; Game 17: boundary-adjacent NE).
2. **The 100-Game Ensemble** (generated via Latin Hypercube Sampling in `data/06_saa_tuned_100_games/`).

All visual presentations adhere strictly to:
- **Tailored Parameter Color**: RIS-Grün (`#009D81` / RGB `[0, 157, 120]`).
- **Universal Default Color**: Warm Orange (`#FF7F0E`).
- **Ground Truth Nash Equilibrium**: Dashed Black (`k--`).
- **Cost Minimization Framework**: Costs $C_i(u) = -J_i(u)$ and axis labels explicitly marked as "Cost" rather than "Payoff".
- **Logarithmic Regret**: $\log_{10}(\text{Regret}_t)$ and $\log_{10}(\text{CumulativeRegret})$.
- **Zero Optuna Artifacts**: Strict focus on MTBO Tailored vs. Universal Default vs. Ground Truth.

---

## 2. Mathematical Formulations & Critique

### 2.1 Why Saltelli/Sobol Sample Size $N = 1024$?
In variance-based global sensitivity analysis (Sobol-Saltelli decomposition), the total variance $V(Y)$ of a model output $Y = f(X_1, \dots, X_d)$ is decomposed into orthogonal ANOVA components:
$$V(Y) = \sum_{i=1}^d V_i + \sum_{i < j} V_{ij} + \dots + V_{12\dots d}$$
The Saltelli (2010) estimator generates two independent $N \times d$ quasi-random base matrices $A$ and $B$, from which $d$ cross-matrices $A_B^{(i)}$ are constructed, requiring:
$$\mathcal{M}_{\text{evals}} = N(d + 2) \quad \text{evaluations per game.}$$
- **Why $N = 1024$**: Sobol sequences are base-2 $(t, m, s)$-nets. They exhibit optimal dyadic stratification across all projections of the unit hypercube $[0, 1)^d$ when $N = 2^m$. At $m = 10$, $N = 2^{10} = 1024$. In static black-box benchmarks, $N = 1024$ is standard to bound estimator error variance.
- **The Scalability Trap**: For dynamic iterative simulations of 5,000 to 20,000 steps, $N = 1024$ for $d = 5$ (`NESeeking5D`) requires $1024 \times 7 = 7,168$ full game trajectories per game $\times 5 = 35,840$ simulations.
- **Sobol Sample Size**: Confirmed at $N = 256 = 2^8$, which strictly preserves the base-2 dyadic grid property of the Sobol sequence. With $T = 5,000$ steps per evaluation, evaluations per game are $256 \times (d + 2)$, running in parallel across CPU cores.
- **Convergence Probe**: Run with fixed **Universal Default Hyperparameters** across all 100 games with `ne_tol = 0.05` and `max_steps = 100_000`.
  - *Methodological Rationale*: Using the constant default hyperparameters eliminates hyperparameter variance as an uncontrolled confounding variable, isolating the pure causal effect of intrinsic game characteristics (action set size, condition number $\kappa$, boundary distance, skewness) on regret and convergence speed.
  - *Censoring & Structural Difficulty*: Games where the residual dither neighborhood exceeds $0.05$ will terminate at $t = 100,000$, serving as an empirical marker of games requiring customized tuning.
- **`NESeeking`**:
  - Game 11: $\min_t \|\hat{u}_t - u^*\| = 0.0271$, steady-state $\approx 0.0371$
  - Game 70: $\min_t \|\hat{u}_t - u^*\| = 0.0032$, steady-state $\approx 0.3130$ (averaging $\approx 0.20$ due to dither bias)
  - Game 97: $\min_t \|\hat{u}_t - u^*\| = 0.4298$, steady-state $\approx 0.4323$
  - Game 40: $\min_t \|\hat{u}_t - u^*\| = 0.0004$, steady-state $\approx 0.0735$
  - Game 17: $\min_t \|\hat{u}_t - u^*\| = 0.0642$, steady-state $\approx 0.0960$
- **`ZOGrandientPlay`**:
  - Game 11: $\min_t \|\mu_t - u^*\| = 0.0382$, steady-state $\approx 0.2598$
  - Game 70: $\min_t \|\mu_t - u^*\| = 0.0019$, steady-state $\approx 0.0766$
  - Game 97: $\min_t \|\mu_t - u^*\| = 0.5867$, steady-state $\approx 0.7526$
  - Game 40: $\min_t \|\mu_t - u^*\| = 0.0137$, steady-state $\approx 0.1370$
  - Game 17: $\min_t \|\mu_t - u^*\| = 0.0021$, steady-state $\approx 0.2930$

**Theoretical Reason**:
1. Extremum Seeking (NESeeking) perturbs actions with persistent sinusoids $a_i \sin(\omega_i t)$. By averaging theory (Frihauf et al., 2012), the state $\hat{u}$ tracks only an $\mathcal{O}(a^2)$ residual neighborhood of $u^*$.
2. ZOGrandientPlay uses Gaussian exploration with diminishing variance $\sigma_t = a / t^{0.25}$ and step size $\gamma_t = 4 / (v t)$, so convergence to $u^*$ is asymptotic ($t \to \infty$); at finite horizon $t \le 20,000$, variance remains $\mathcal{O}(0.01 - 0.2)$.
3. **Fatal Vulnerability**: If $\epsilon = 10^{-6}$ or $10^{-3}$ is chosen, **0% of games will stop early**, rendering "steps to converge" trivial and identical to `MAX_STEPS`.
4. **Resolution**:
   - Provide an explicit tolerance $\epsilon_{\text{tol}} \in \{0.05, 0.10\}$ (or relative tolerance $\frac{\|\hat{u}_t - u^*\|}{\|u^*\|} < 0.10$).
   - Impose a safety cap $T_{\max} = 20,000$ steps. If an algorithm does not reach $\epsilon_{\text{tol}}$ by $T_{\max}$, record $T_{\max}$ and report convergence rate ($P(\text{converged})$).

### 2.3 Cost Minimization Equivalence
In the paper:
$$C_i(u) = -J_i(u)$$
Since the single-player best response satisfies:
$$u_i^{\text{BR}}(u_{-i}) = \arg\min_{u_i \in \mathcal{A}_i} C_i(u_i, u_{-i}) = \arg\max_{u_i \in \mathcal{A}_i} J_i(u_i, u_{-i})$$
The instantaneous regret in the cost framework:
$$R_i(t) = C_i(u_i(t), u_{-i}(t)) - C_i(u_i^{\text{BR}}(u_{-i}(t)), u_{-i}(t)) = J_i(u_i^{\text{BR}}(u_{-i}(t)), u_{-i}(t)) - J_i(u_i(t), u_{-i}(t)) \ge 0$$
is strictly invariant. Thus:
1. Regret computations remain identical.
2. In all trajectory plots, plot $-J_i(u)$ and label the y-axis as "Cost $C_i(u)$".
3. In representative game equations, present costs with inverted signs:
   $$C_i(u) = \frac{1}{2} u_i^T Q_{ii}^{\text{cost}} u_i + u_i^T \sum_{j \neq i} Q_{ij}^{\text{cost}} u_j + (r_i^{\text{cost}})^T u_i + c_i^{\text{cost}}$$
   where $Q^{\text{cost}} = -H$ and $r^{\text{cost}} = -d$.

---

## 3. Features & Interface Modifications

Direct modification of `payoffnash/ne_seeking.py` and `payoffnash/zo_gradient_play.py` without wrappers:

### 3.1 Interface Update in `NEEstimator` (`common.py`)
```python
    @abstractmethod
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
        pass
```

### 3.2 Implementation in `NESeeking` (`ne_seeking.py`)
```python
            if track_ne:
                dist_ne = float(np.linalg.norm(flat_u_hat - flat_true_ne))
                if dist_ne <= ne_tol:
                    break
```

### 3.3 Implementation in `ZOGrandientPlay` (`zo_gradient_play.py`)
```python
            if track_ne:
                dist_ne = float(np.linalg.norm(flat_state - flat_true_ne))
                if dist_ne <= ne_tol:
                    break
```

Both classes return `EstimateHistory` truncated to the step of convergence $t$, so `steps_to_converge = len(history.action_history) - 1`.

---

## 4. Notebook Structure (`results.ipynb`)

All artifacts and figures are co-located in `data/08_final_results/`.

### Section 1: Executive Summary & Theoretical Context
- Cost minimization formulation ($C_i = -J_i$).
- Algorithmic properties of zeroth-order tracking and residual dither balls.

### Section 2: 5 Representative Games — Trajectory & Regret Analysis
- For each of the 5 representative games (`[11, 70, 97, 40, 17]`):
  - Action History $u_i(t)$ vs. Time $t$.
  - State History $\hat{u}_i(t)$ vs. Time $t$.
  - Instantaneous Regret $\log_{10}(R_i(t))$ vs. Time $t$.
  - Instantaneous Cost $C_i(u(t))$ vs. Time $t$.
- Styles:
  - MTBO Tailored: RIS-Grün (`#009D81`, solid line).
  - Universal Default: Warm Orange (`#FF7F0E`, solid line).
  - True Nash Equilibrium $u^*$: Dashed Black (`k--`).
  - Optuna curves: Completely omitted.

### Section 3: Sobol Sensitivity Analysis (NESeeking & ZOGrandientPlay)
- Parameters:
  - `NESeeking5D`: $k_0, k_1, a_0, a_1, dt$ ($d = 5$).
  - `ZOGrandientPlay`: $a, v, \epsilon$ ($d = 3$).
- Design: Saltelli sampling with $N = 128$, $T = 5,000$ steps per evaluation.
- Execution: Parallelized across CPU cores via `joblib.Parallel` or `concurrent.futures`.
- Plots: Bar charts of First-Order ($S_1$) and Total-Order ($S_T$) Sobol indices across the 5 representative games using RIS-Grün monochromatic color gradient.

### Section 4: 100-Game Tunability Correlations (4 Pairs of Point Plots)
- Compute 4 defining game characteristics for all 100 games:
  1. **Action Set Size**: Box volume $\prod_{i} (\bar{a}_i - \underline{a}_i)$ or diameter $\|\bar{a} - \underline{a}\|_2$.
  2. **Condition Number $\kappa$**: $\frac{\lambda_{\max}(Q + Q^T)}{\lambda_{\min}(Q + Q^T)}$.
  3. **Closeness of NE to Boundary**: $\min_{i,k} \min(u^*_{i,k} - \underline{a}_{i,k}, \bar{a}_{i,k} - u^*_{i,k})$.
  4. **Coupling Skewness**: Frobenius ratio $\frac{\|Q_{ij}\|_F}{\|Q_{ji}\|_F}$ or asymmetry $\|Q - Q^T\|_F / \|Q + Q^T\|_F$.
- 4 pairs of scatter plots (NESeeking & ZOGrandientPlay):
  - X-axis: Defining characteristic.
  - Y-axis: Tunability Gain $U(g) = \text{Score}(\theta_{\text{def}}) - \text{Score}(\theta_{\text{tailored}})$.
  - Point styling: RIS-Grün (`#009D81`) with alpha transparency and OLS trendline.
  - Statistical annotation: Pearson correlation coefficient $r$, $p$-value, and Spearman rank coefficient $\rho$.

### Section 5: 100-Game Convergence & Stopping Horizon Analysis
- Simulation: Run all 100 games until $\|\hat{u}_t - u^*\| \le \epsilon_{\text{tol}}$ or $t = T_{\max} = 20,000$.
- Record:
  - Total Regret at termination.
  - Number of iterations to converge.
- Statistical Table: Mean and variance of Total Regret and Iteration Steps across Default vs. Tailored parameters for both algorithms.
- 4 pairs of scatter plots: Total Regret (or Iteration Steps) vs. the 4 defining characteristics.

---

## 5. Acceptance & Validation Criteria
1. **Zero Optuna Artifacts**: Neither Optuna code nor Optuna lines appear anywhere in `results.ipynb`.
2. **Color Conformity**: All tailored plots use RIS-Grün (`#009D81`); all default plots use `#FF7F0E`.
3. **Execution Safety**: Sobol sensitivity runs at $T = 5,000$, $N = 128$ in parallel; total notebook execution completes cleanly without OOM or kernel lockup.
4. **Code Quality**: All modifications to `payoffnash/ne_seeking.py` and `payoffnash/zo_gradient_play.py` pass the existing 41-test suite in `tests/` without regressions.
5. **Artifact Co-location**: All generated plots and summary tables saved to `data/08_final_results/`.
