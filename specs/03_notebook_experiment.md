# Experiment Specification: Large-Scale Algorithmic Benchmark & Comparative Analysis (`analysis.ipynb`)

**Context:** This document specifies the empirical experiment to be executed in the interactive Jupyter notebook [`analysis.ipynb`](file:///home/bruno/Documents/PayoffBasedNEComparison/analysis.ipynb). The goal is to benchmark and compare two payoff-based zeroth-order Nash Equilibrium seeking algorithms—Zero-Order Gradient Play ([`ZOGrandientPlay`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/zo_gradient_play.py)) and Continuous-Time Extremum Seeking ([`NESeeking5D`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/ne_seeking.py))—across an ensemble of 100 synthetic quadratic games using Multi-Task Bayesian Optimization ([`MetaParamOptimization`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/mtbo.py)).

---

### 1. Game Ensemble Generation ($M = 100$)

Using `QuadraticGameGenerator` from [`payoffnash/game_ensemble.py`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/game_ensemble.py):
* **Number of games ($M$):** $100$.
* **Game dimensions:** $N = 2$ players, $m = 1$ action dimension per player ($Nm = 2$).
* **Strong monotonicity margin ($\alpha$):** $0.05$ (ensuring strict negative definiteness $\lambda_{\max}(H_{\text{sym}}) \le -0.05$).
* **Payoff coefficient interval:** $[-1.0, 1.0]$.
* **Action space boundaries:** $lb \in [-1.0, -0.2]$, $ub \in [0.2, 1.0]$.
* **Interior Equilibrium:** `enforce_interior=True` (guaranteeing that the true theoretical unconstrained Nash equilibrium $u^* = -H^{-1}d$ lies strictly within the interior of the action box $lb < u^* < ub$).
* **Random Seed:** $14062003$ for strict reproducibility.

---

### 2. Algorithmic Configurations, Search Spaces & Generalized Regret Cost

#### A. Generalized Logarithmic Total Regret Cost Function
For any game in the ensemble with symmetric/asymmetric coupling $H \in \mathbb{R}^{2 \times 2}$, linear terms $d \in \mathbb{R}^2$, and bounded action box $[\text{lb}, \text{ub}]$, player $i$'s unconstrained best response to opponent actions $u_{-i}$ is:
$$u_i^{\text{unc}}(u_{-i}) = -H_{ii}^{-1} \left(H_{ij} u_j + d_i\right)$$
Projected onto the action interval:
$$u_i^*(u_{-i}) = \text{clip}\left(u_i^{\text{unc}}(u_{-i}), \; \text{lb}_i, \; \text{ub}_i\right)$$
The instantaneous regret at step $t$ is the payoff lost relative to the best response:
$$R_i(t) = J_i(u_i^*(u_{-i}(t)), u_{-i}(t)) - J_i(u_i(t), u_{-i}(t)) \ge 0$$
Over a simulation length of $T = 20,000$ steps ($20,001$ history states), MTBO minimizes the logarithmic total systemic regret:
$$\text{Cost}(g, \text{history}) = \log_{10}\left(\max\left(\sum_{t=0}^T \sum_{i=1}^N R_i(t), \; 10^{-12}\right)\right)$$

#### B. Zero-Order Gradient Play (`ZOGrandientPlay`)
* **Reference:** Tatarenko & Kamgarpour (2022).
* **Estimation Type:** Strictly `"one-point"` (single payoff query per player per step).
* **Scale-Invariant Search Space (2 Continuous Dimensions):**
  - $v \in [0.01, 100.0]$: Gradient step-size gain.
  - $a \in [0.005, 2.0]$: Dimensionless relative perturbation ratio ($0.5\%$ to $200\%$ of bottleneck bound width).
* **Scale Invariance (`scaled_params`):**
  - `scaled_params = {"a": "min_box_width"}`
  - Effective scalar amplitude: $a_{\text{eff}} = a \cdot \min_{i} (ub_i - lb_i)$.
* **Fixed Parameters:**
  - `epsilon`: $10^{-4}$ (decay stabilization parameter).
  - `estimation_type`: `"one-point"`.
* **Simulation Length:** `num_steps = 20000`.

#### C. Continuous-Time Nash Equilibrium Seeking (`NESeeking5D`)
* **Reference:** Frihauf, Krstic, & Basar (2012) and Dimension Reduction Study (`specs/06_neseeking_dimension_reduction_study.md`).
* **Scale-Invariant Search Space (5 Continuous Dimensions):**
  - Integration gains: $k[0,0], k[1,0] \in [0.01, 100.0]$
  - Relative perturbation ratios: $a[0,0], a[1,0] \in [0.005, 2.0]$ ($0.5\%$ to $200\%$ of each player's action bound width).
  - Base dither frequency: $\omega_1 \in [1.0, 50.0]$
* **Scale Invariance (`scaled_params`):**
  - `scaled_params = {"a[0,0]": "player_box_width", "a[1,0]": "player_box_width"}`
  - Effective amplitudes: $a_i = a[i,0] \cdot (ub_i - lb_i)$.
* **Coupled Degrees of Freedom (Option A):**
  $$\omega_2 = 1.35 \cdot \omega_1, \qquad \omega_{h1} = 0.20 \cdot \omega_1, \qquad \omega_{h2} = 0.20 \cdot \omega_2$$
* **Fixed Parameters:**
  - `dt`: $0.001$ (forward Euler integration timestep).
* **Simulation Length:** `num_steps = 20000`.

---

### 3. Batched MTBO Execution Architecture ($4 \times 25$ Batches)

To completely eliminate exact Gaussian Process memory exhaustion ($\mathcal{O}(N^2)$ scaling where $N = 25,600 \to 25\text{ GB}$), the 100 games are split into **4 sequential batches of 25 games each**:
- **Batch 1:** Games 0 to 24
- **Batch 2:** Games 25 to 49
- **Batch 3:** Games 50 to 74
- **Batch 4:** Games 75 to 99

#### A. Optimization Budget per Batch
* **Initial Quasi-Random Phase ($n_{\text{init}}$):**
  - `ZOGrandientPlay` (2D): $n_{\text{init}} = 16$ (Sobol block of $16 \times 25 = 400$ training points, kernel matrix size $1.28\text{ MB}$).
  - `NESeeking5D` (5D): $n_{\text{init}} = 32$ (Sobol block of $32 \times 25 = 800$ training points, kernel matrix size $5.12\text{ MB}$).
* **Stricter Early Stopping Tolerances:**
  - `eps_ei`: $10^{-5}$ ($0.00001$), analytic Expected Improvement floor threshold.
  - `patience`: $10$ iterations window for detecting incumbent stagnation.
  - `tol_stagnation`: $10^{-5}$ minimum required improvement over the patience window.
  - `max_bo_steps`: $200$ safety ceiling.
* **Option B Continuous Surrogate Polish:**
  - At the end of each batch, continuous L-BFGS-B polish runs on the surrogate posterior mean $\mu(x)$ within local trust regions for all 25 tasks and the batch default.
  - Verified in a single parallel verification batch across worker processes.

#### B. Option B: Global Consensus Default Determination
1. Each of the 4 batches produces 25 game-specific tailored optima $\theta_j^*$ and 1 batch-candidate default $\hat{\theta}_{\text{def}}^{(b)}$ ($b=1,\dots,4$).
2. Aggregate the candidate pool:
   $$\mathcal{C} = \left\{\hat{\theta}_{\text{def}}^{(1)}, \hat{\theta}_{\text{def}}^{(2)}, \hat{\theta}_{\text{def}}^{(3)}, \hat{\theta}_{\text{def}}^{(4)}, \; \bar{\theta}_{\text{mean}}\right\}$$
   where $\bar{\theta}_{\text{mean}} = \frac{1}{4} \sum_{b=1}^4 \hat{\theta}_{\text{def}}^{(b)}$.
3. In a single parallel verification pass (evaluating the 5 candidates across all 100 games in parallel), identify the single configuration that minimizes the ensemble mean regret across all 100 games:
   $$\theta_{\text{global\_def}}^* = \arg\min_{\theta \in \mathcal{C}} \frac{1}{100} \sum_{j=1}^{100} \text{Loss}(G_j; \theta)$$
4. Compute the **Global Tunability Index** across all 100 games against this single, unified universal default:
   $$\text{Tunability}_j = \text{Loss}(G_j; \theta_{\text{global\_def}}^*) - \text{Loss}(G_j; \theta_j^*) \ge 0$$

---

### 4. Automated Selection & Optuna Baseline on 5 Diverse Representative Games

#### A. Geometric Selection Across the Entire 100 Games
Selected programmatically from the entire 100-game ensemble using objective criteria:
1. **Extreme Ill-Conditioned:** Maximizes $\kappa(H_{\text{sym}}) = \lambda_{\max} / \lambda_{\min}$.
2. **Well-Behaved / Isotropic:** Minimizes condition number $\kappa(H_{\text{sym}}) \approx 1.0$.
3. **Narrow Action Space:** Minimizes bounding box volume $\prod (ub - lb)$.
4. **Highly Asymmetric Coupling:** Maximizes anti-symmetric norm $\|H - H^T\|_F$.
5. **Boundary-Adjacent Equilibrium:** Minimizes distance of true NE $u^*$ to bounds.

#### B. Per-Game Optuna Baseline Optimization
Run an independent Optuna TPE study on each of the 5 representative games:
* **Algorithms:** Evaluated for both `ZOGrandientPlay` (2D) and `NESeeking` (5D).
* **Trials:** $100$ trials per game.
* **Objective:** Direct single-game simulation minimizing $\log_{10}(\text{Total Regret})$ over 20,000 steps.

---

### 5. Action Trajectory Visualization Architecture

For each of the 5 representative games, generate a 2x2 subplot trajectory comparison over 20,000 steps:
* **Top-Left:** Player 1 Action $u_1(t)$ comparing MTBO Tailored vs. Optuna vs. True NE $u_1^*$.
* **Top-Right:** Player 1 Action $u_1(t)$ comparing MTBO Global Consensus Default vs. True NE $u_1^*$.
* **Bottom-Left:** Player 2 Action $u_2(t)$ comparing MTBO Tailored vs. Optuna vs. True NE $u_2^*$.
* **Bottom-Right:** Player 2 Action $u_2(t)$ comparing MTBO Global Consensus Default vs. True NE $u_2^*$.

---

### 6. Quantitative Analysis & Reporting

At the conclusion of the notebook:
1. **Representative Games Comparison Table:**
   - Display a Pandas DataFrame comparing achieved total regret across:
     - Optuna Tailored Optima
     - MTBO Tailored Optima $\theta_j^*$
     - MTBO Global Consensus Default $\theta_{\text{global\_def}}^*$
2. **Ensemble Statistics Across All 100 Games:**
   - Report mean, median, standard deviation, and maximum regret across all 100 games.
   - Report Tunability Index distribution ($\Delta \text{Loss}_j$) and 95% confidence intervals.
   - Run paired $t$-test and Wilcoxon signed-rank test on $H_0: \mathbb{E}[\Delta \text{Loss}] = 0$.
3. **Computational Efficiency & Speedup Benchmarking:**
   - Record exact wall-clock training durations: MTBO total execution time ($T_{\text{MTBO}}$) vs. extrapolated 100-game Optuna runtime ($T_{\text{Optuna, 100}}^{\text{extrap}} = 20 \times T_{\text{Optuna, 5}}$).
   - Speedup factor and total query efficiency.
4. **Automated Morning Markdown Report (`100_games_comparative_report.md`):**
   - Automatically compile and write a structured executive report to `data/05_scaled_100_games_regret/100_games_comparative_report.md`.
