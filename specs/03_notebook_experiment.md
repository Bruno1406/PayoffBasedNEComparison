# Experiment Specification: Large-Scale Algorithmic Benchmark & Comparative Analysis (`analysis.ipynb`)

**Context:** This document specifies the empirical experiment to be executed in the interactive Jupyter notebook [`analysis.ipynb`](file:///home/bruno/Documents/PayoffBasedNEComparison/analysis.ipynb). The goal is to benchmark and compare two payoff-based zeroth-order Nash Equilibrium seeking algorithms—Zero-Order Gradient Play ([`ZOGrandientPlay`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/zo_gradient_play.py)) and Extremum Seeking / Nash Equilibrium Seeking ([`NESeeking`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/ne_seeking.py))—across an ensemble of 100 synthetic quadratic games using Multi-Task Bayesian Optimization ([`MetaParamOptimization`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/mtbo.py)).

---

### 1. Game Ensemble Generation ($M = 100$)

Using `QuadraticGameGenerator` from [`payoffnash/game_ensemble.py`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/game_ensemble.py):
* **Number of games ($M$):** $100$.
* **Game dimensions:** $N = 2$ players, $m = 1$ action dimension per player ($Nm = 2$).
* **Strong monotonicity margin ($\alpha$):** $0.05$ (ensuring strict negative definiteness $\lambda_{\max}(H_{\text{sym}}) \le -0.05$).
* **Payoff coefficient interval:** $[-1.0, 1.0]$.
* **Action space boundaries:** $lb \in [-1.0, -0.2]$, $ub \in [0.2, 1.0]$.
* **Interior Equilibrium:** `enforce_interior=True` (guaranteeing that the true theoretical unconstrained Nash equilibrium $u^* = -H^{-1}d$ lies strictly within the interior of the action box $lb < u^* < ub$).
* **Random Seed:** $42$ for strict reproducibility.

---

### 2. Algorithmic Configurations & Metaparameter Search Spaces

#### A. Zero-Order Gradient Play (`ZOGrandientPlay`)
* **Reference:** Tatarenko & Kamgarpour (2022).
* **Estimation Type:** Strictly `"one-point"` (single payoff query per player per step).
* **Search Space:**
  - $v \in [0.5, 10.0]$: Gradient step-size gain.
  - $a \in [0.01, 1.0]$: Random exploration perturbation variance $\sigma_k = \frac{a}{(k+1)^{1/6}}$.
* **Fixed Parameters:**
  - `estimation_type`: `"one-point"`.
  - `epsilon`: $10^{-4}$ (decay stabilization parameter).
* **Simulation Length:** $N_{\text{iter}} = 3000$ steps.
* **Loss Metric Window:** $K = 50$ steps moving average equilibrium error.

#### B. Continuous-Time Nash Equilibrium Seeking (`NESeeking`)
* **Reference:** Frihauf, Krstic, & Basar (2012).
* **Search Space:**
  - $k \in [0.1, 5.0]$: Integrator gain (broadcast across players to shape $(2, 1)$).
  - $a \in [0.01, 0.5]$: Dither perturbation amplitude (broadcast across players to shape $(2, 1)$).
* **Fixed Parameters:**
  - `omega`: $\begin{bmatrix} 25.0 \\ 31.0 \end{bmatrix}$ (distinct dither perturbation frequencies for players 1 and 2).
  - `phi`: $\begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$ (phase shifts).
  - `dt`: $0.005$ (forward Euler integration timestep).
* **Simulation Length:** $N_{\text{iter}} = 3000$ steps.
* **Loss Metric Window:** $K = 50$ steps moving average equilibrium error.

---

### 3. MTBO Execution Plan & 2–3 Hour Overnight Budgeting

Using active task-selecting MTBO ([`MetaParamOptimization`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/mtbo.py)):
* **Active Task Selection:** In each BO iteration, Swersky's two-step heuristic queries a single game $t^* \in \{0, \dots, 99\}$ and hyperparameter $x^*$, running **only 1 simulation** per BO step.
* **Initial Quasi-Random Phase:** $n_{\text{init}} = 100$ points via Latin Hypercube Sampling, systematically paired with tasks $t_i = i \pmod{100}$ so that each of the 100 games has at least 1 evaluated baseline for the initial task covariance kernel.
* **Sequential BO Phase:** $n_{\text{bo_steps}} = 500$ sequential queries per algorithm.
* **Runtime Estimation:**
  - Simulation cost ($N_{\text{iter}} = 3000$): $\approx 0.10\text{ s}$ (`ZOGrandientPlay`), $\approx 0.25\text{ s}$ (`NESeeking`).
  - GP surrogate fit & acquisition: $\approx 1.5 - 3.0\text{ s}$ per query.
  - Per query total: $\approx 2.0 - 3.2\text{ s}$.
  - Total queries per algorithm: $100 + 500 = 600$ queries.
  - Estimated duration per algorithm: $600 \times 2.5\text{ s} \approx 1500\text{ s} \approx 25 - 40\text{ minutes}$ per algorithm.
  - Total combined runtime for both algorithms: $\approx 1.2 - 2.0\text{ hours}$ (safely within the 2–3 hour target window).
* **Artifact Export:**
  - `zo_optimal_metaparams.csv`
  - `ne_optimal_metaparams.csv`

---

### 4. Automated Selection of 5 Diverse Representative Games

Because plotting 100 games produces visual clutter, 5 representative games must be selected programmatically from the 100-game ensemble using objective geometric and algebraic criteria:

1. **Game 1 (Extreme Ill-Conditioned):**
   - Criterion: Maximizes the condition number of the symmetrized Hessian:
     $$\kappa(H_{\text{sym}}) = \frac{\lambda_{\max}(H_{\text{sym}})}{\lambda_{\min}(H_{\text{sym}})}$$
   - Role: Tests algorithm robustness under severe directional curvature distortion and slow convergence modes.
2. **Game 2 (Well-Behaved / Isotropic):**
   - Criterion: Minimizes condition number $\kappa(H_{\text{sym}}) \approx 1.0$.
   - Role: Evaluates baseline convergence in an ideal, uniformly curved game space.
3. **Game 3 (Narrow Action Space):**
   - Criterion: Minimizes the bounding box hypervolume $\prod_{p=1}^N \prod_{d=1}^m (ub_{p,d} - lb_{p,d})$.
   - Role: Stresses boundary projection dynamics and boundary-clipping oscillations.
4. **Game 4 (Highly Asymmetric Cross-Coupling):**
   - Criterion: Maximizes the anti-symmetric off-diagonal Frobenius norm:
     $$\|H - H^T\|_F$$
   - Role: Stresses rotational dynamics, non-conservative pseudogradient flow, and inter-player coupling.
5. **Game 5 (Boundary-Adjacent Equilibrium):**
   - Criterion: Minimizes distance of the true Nash equilibrium $u^*$ to the action boundaries:
     $$\min_{p, d} \min(u_{p, d}^* - lb_{p, d}, ub_{p, d} - u_{p, d}^*)$$
   - Role: Tests if exploration noise causes premature boundary saturation or bias near constraints.

---

### 5. Action-Only Visualization Architecture

In [`analysis.ipynb`](file:///home/bruno/Documents/PayoffBasedNEComparison/analysis.ipynb):
* **Plot Actions Only:** Exclude payoff curves completely to prevent clutter.
* **Layout:** For each of the 5 selected representative games, generate a dedicated 2x2 subplot figure:
  - **Top-Left (Row 1, Col 1):** Player 1 Action $u_1(t)$ under **Game-Specific Optimum $\theta_j^*$**.
    - Solid blue trajectory: $u_1(t)$.
    - Dashed black line: True Nash Equilibrium $u_1^*$.
  - **Top-Right (Row 1, Col 2):** Player 1 Action $u_1(t)$ under **Universal Default Optimum $\theta_{\text{default}}^*$**.
    - Solid orange trajectory: $u_1(t)$.
    - Dashed black line: True Nash Equilibrium $u_1^*$.
  - **Bottom-Left (Row 2, Col 1):** Player 2 Action $u_2(t)$ under **Game-Specific Optimum $\theta_j^*$**.
    - Solid blue trajectory: $u_2(t)$.
    - Dashed black line: True Nash Equilibrium $u_2^*$.
  - **Bottom-Right (Row 2, Col 2):** Player 2 Action $u_2(t)$ under **Universal Default Optimum $\theta_{\text{default}}^*$**.
    - Solid orange trajectory: $u_2(t)$.
    - Dashed black line: True Nash Equilibrium $u_2^*$.
* **Title & Annotation:** Include game category, condition number $\kappa(H_{\text{sym}})$, distance to boundary, and final average error values in the figure title or subtitle.

---

### 6. Quantitative Analysis & Reporting

At the conclusion of the notebook:
1. **Comparative Summary Table:**
   - Display a Pandas DataFrame containing:
     - Game Index & Category (Ill-conditioned, Isotropic, Narrow, Asymmetric, Boundary-adjacent).
     - True NE coordinates $[u_1^*, u_2^*]$.
     - Game-Specific Optimum $\theta_j^*$ and achieved error $g_j(\theta_j^*)$.
     - Universal Default Optimum $\theta_{\text{default}}^*$ and achieved error $g_j(\theta_{\text{default}}^*)$.
     - Tunability Gain: $\Delta g_j = g_j(\theta_{\text{default}}^*) - g_j(\theta_j^*)$.
2. **Ensemble Statistics:**
   - Report mean, median, standard deviation, and maximum equilibrium distance across all 100 games under default vs. tailored parameters.
3. **Artifact Summary Report:**
   - Write a detailed report markdown file comparing both algorithms to the artifacts directory.
