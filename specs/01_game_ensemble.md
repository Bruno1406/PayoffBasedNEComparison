# Implementation Specification: Quadratic Game Ensemble Generator

**Context:** This module is part of a bachelor thesis evaluating the tunability and robustness of payoff-based (zeroth-order) Nash equilibrium seeking algorithms. The immediate deliverable is a synthetic game generation module that creates an unbiased, mathematically valid ensemble of strongly monotone quadratic games. These games act as the benchmark distribution for downstream Multi-Task Bayesian Optimization (MTBO) to calculate the Tunability Index.

---

### 1. Module Objectives

Implement a class `QuadraticGameGenerator` responsible for procedurally generating an ensemble of synthetic, strongly monotone quadratic games. The generator uses Latin Hypercube Sampling (LHS) via `scipy.stats.qmc.LatinHypercube` to draw payoff coefficients and action space boundaries from decoupled intervals, strictly validating each candidate game to guarantee a unique, stable Nash equilibrium (strong monotonicity).

---

### 2. Input Parameters

The generator class/method accepts the following initialization parameters:

* **Number of players ($N$):** Integer representing the number of agents ($N \ge 2$).
* **Action space dimension per player ($m$):** Integer representing the dimension of each player's action vector (default: `1`).
* **Payoff Coefficients Interval:** Default to `[-1.0, 1.0]`. Range for sampling quadratic terms, cross-coupling terms, and linear terms.
* **Action Space Lower Bound Interval:** Default to `[-1.0, 0.0)`. The lower bounds ($lb_i$) of the hypercube constraining each player's action set are sampled from this strictly negative range to guarantee the origin is included.
* **Action Space Upper Bound Interval:** Default to `(0.0, 1.0]`. The upper bounds ($ub_i$) of the hypercube constraining each player's action set are sampled from this strictly positive range.
* **Ensemble Size:** Integer specifying the total number of valid game objects to generate and return.
* **Strong Monotonicity Margin ($\alpha$):** Float $\alpha > 0$ (default: `1e-3`), ensuring strict negative definiteness with a minimal stability margin.
* **Enforce Interior Equilibrium:** Boolean (default: `False`). When enabled, guarantees that the theoretical unconstrained Nash equilibrium $u^*$ lies strictly in the interior of the action space ($lb < u^* < ub$).
* **Random Seed:** Optional integer seed for reproducible LHS generation.

---

### 3. Mathematical Formulation & Validation Logic

The generation follows an efficient propose-and-verify batching loop:

#### A. LHS Proposing
1. Use `scipy.stats.qmc.LatinHypercube(d=...)` to sample batches of candidate parameters in $[0, 1)$.
2. Scale each parameter slice independently to its target interval:
   - Payoff coefficients $\to [-1.0, 1.0]$
   - Action lower bounds $lb_i \to [-1.0, 0.0)$
   - Action upper bounds $ub_i \to (0.0, 1.0]$

#### B. Quadratic Payoff Structure
For player $i$, the payoff function $J_i(u)$ is:
$$J_i(u) = \frac{1}{2} u_i^T D_{ii}^i u_i + u_i^T \sum_{j \neq i}^N D_{ij}^i u_j + (d_i^i)^T u_i + c^i(u_{-i})$$

where:
- $u_i \in \mathbb{R}^m$ is player $i$'s action vector, and $u = [u_1^T, \dots, u_N^T]^T \in \mathbb{R}^{Nm}$.
- $D_{ii}^i \in \mathbb{R}^{m \times m}$ is symmetric, governing self-curvature (concavity).
- $D_{ij}^i \in \mathbb{R}^{m \times m}$ governs the cross-coupling sensitivity of player $i$ with respect to player $j$'s action.
- $d_i^i \in \mathbb{R}^m$ is the linear payoff term for player $i$.
- $c^i(u_{-i})$ represents optional player externalities (terms not depending on $u_i$, which do not affect player $i$'s best-response or marginal payoff).

The individual marginal payoff (gradient) for player $i$ is:
$$\nabla_{u_i} J_i(u) = D_{ii}^i u_i + \sum_{j \neq i}^N D_{ij}^i u_j + d_i^i$$

Stacking these gradients yields the **game pseudogradient**:
$$G(u) = \begin{bmatrix} \nabla_{u_1} J_1(u) \\ \vdots \\ \nabla_{u_N} J_N(u) \end{bmatrix} = H u + d$$

where $H \in \mathbb{R}^{Nm \times Nm}$ is the game Jacobian (game Hessian), structured from blocks $H_{ij} = D_{ij}^i$, and $d = [(d_1^1)^T, \dots, (d_N^N)^T]^T \in \mathbb{R}^{Nm}$.

#### C. Strong Monotonicity Validation (via Pseudogradient Hessian)
Rather than checking conservative row-sum diagonal dominance, the generator validates strong monotonicity directly via the symmetric part of the game Hessian:
$$H_{sym} = \frac{H + H^T}{2}$$

**Condition:**
The game is accepted if and only if:
$$\lambda_{\max}(H_{sym}) \le -\alpha < 0 \quad (\text{i.e. } H_{sym} \preceq -\alpha I)$$

**Mathematical Justification:**
1. **Tatarenko & Kamgarpour (2022) Compatibility:** Directly satisfies Assumption 1 (strong monotonicity of the payoff game operator), ensuring convergence of `ZOGrandientPlay`.
2. **Frihauf et al. (2012) Compatibility:** By Bendixson's theorem on matrix spectra, $\text{Re}(\lambda(H)) \le \lambda_{\max}(H_{sym}) \le -\alpha < 0$. Therefore, $H$ is strictly Hurwitz, guaranteeing asymptotic stability of the continuous-time/Euler dynamics in `NESeeking`.
3. **Multi-Dimensional Invariance:** The condition applies identically to scalar actions ($m=1$) and multi-dimensional actions ($m > 1$) without requiring ad-hoc block-matrix dominance definitions.
4. **Feasible Sampling Efficiency:** Yields acceptance rates orders of magnitude higher than strict diagonal dominance (e.g. >50x higher at $N=3$, feasible at $N=4$ via vectorized NumPy).

#### D. Interior Nash Equilibrium Validation
If `enforce_interior=True`:
1. Calculate the analytical unconstrained Nash equilibrium:
   $$u^* = -H^{-1} d$$
2. Verify that $u^*$ lies strictly inside the sampled action box:
   $$lb_{i, k} < u_{i, k}^* < ub_{i, k} \quad \forall i \in \{1,\dots,N\}, k \in \{1,\dots,m\}$$
3. Candidate games where $u^*$ falls on or outside the action boundaries are discarded.

#### E. Rejection & Batch Sampling Strategy
Because LHS parameter draws may be discarded during validation:
- The generator draws candidate batches of size $K = \max(200, 10 \times \text{ensemble\_size})$.
- Filtering ($H_{sym}$ eigenvalue computation and boundary check) is vectorized across the batch.
- Valid games are collected until the requested `ensemble_size` is fulfilled.

---

### 4. Output Specification & Architecture

#### Return Type: `list[QuadraticGame]`
The generator returns a list of `QuadraticGame` objects. To maintain backwards compatibility with existing codebase estimators, `QuadraticGame` inherits from [`Game`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/common.py#L9):

```python
@dataclass
class QuadraticGame(Game):
    H: np.ndarray                   # (Nm, Nm) game Hessian matrix
    d: np.ndarray                   # (Nm,) linear payoff vector
    c: np.ndarray                   # (N,) constants / externality offsets
    nash_equilibrium: np.ndarray    # (Nm,) exact theoretical NE u* = -H^{-1}d
    optimal_payoffs: list[float]    # Theoretical payoff values J_i(u*)
```

- Each player's callable `payoff_function` in `payoff_functions` is automatically generated from $H$, $d$, and $c$, accepting flat actions $(u_1, \dots, u_{Nm})$ matching the signature required by [`Game`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/common.py#L9).
- Exposing `nash_equilibrium` and `optimal_payoffs` allows downstream MTBO and evaluation scripts (e.g., in `analysis.ipynb`) to compute cumulative regret directly and analytically without numerical approximations.

---

### 5. Validation Test Suite Plan

The module must include a dedicated test suite (`tests/test_game_ensemble.py`):
1. **Strong Monotonicity Test:** Verify that for every generated game, all eigenvalues of $H_{sym}$ are $\le -\alpha$, and all eigenvalues of $H$ have strictly negative real parts (Hurwitz).
2. **Boundary & Origin Feasibility:** Verify that $lb_i < 0 < ub_i$, ensuring the origin $0 \in [lb_i, ub_i]$.
3. **Interior Nash Equilibrium Test:** Verify that $lb < u^* < ub$ for all generated games when `enforce_interior=True`.
4. **Payoff Consistency Check:** Confirm that evaluating the callable `payoff_functions` matches the matrix multiplication $\frac{1}{2} u^T \bar{D}^i u + d^i u + c^i$.
5. **Estimator Convergence Test:** Run a small instance with [`NESeeking`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/ne_seeking.py#L13) and [`ZOGrandientPlay`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/zo_gradient_play.py#L13) on a generated game to verify that both algorithms converge toward the theoretical `nash_equilibrium`.
6. **Reproducibility Test:** Verify that supplying the same `seed` generates an identical ensemble of games.