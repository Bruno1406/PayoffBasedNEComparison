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

### 2. Algorithmic Configurations, Search Spaces & Generalized Regret Cost

#### A. Generalized Logarithmic Total Regret Cost Function
For any game in the ensemble with symmetric/asymmetric coupling $H \in \mathbb{R}^{2 \times 2}$, linear terms $d \in \mathbb{R}^2$, and bounded action box $[\text{lb}, \text{ub}]$, player $i$'s unconstrained best response to opponent actions $u_{-i}$ is:
$$u_i^{\text{unc}}(u_{-i}) = -H_{ii}^{-1} \left(H_{ij} u_j + d_i\right)$$
Projected onto the action interval:
$$u_i^*(u_{-i}) = \text{clip}\left(u_i^{\text{unc}}(u_{-i}), \; \text{lb}_i, \; \text{ub}_i\right)$$
The instantaneous regret at step $t$ is the payoff lost relative to the best response:
$$R_i(t) = J_i(u_i^*(u_{-i}(t)), u_{-i}(t)) - J_i(u_i(t), u_{-i}(t)) \ge 0$$
Over a simulation length of $T = 10,000$ steps ($10,001$ history states), MTBO minimizes the logarithmic total systemic regret:
$$\text{Cost}(g, \text{history}) = \log_{10}\left(\max\left(\sum_{t=0}^T \sum_{i=1}^N R_i(t), \; 10^{-12}\right)\right)$$

#### B. Zero-Order Gradient Play (`ZOGrandientPlay`)
* **Reference:** Tatarenko & Kamgarpour (2022).
* **Estimation Type:** Strictly `"one-point"` (single payoff query per player per step).
* **Search Space (2 Continuous Dimensions):**
  - $v \in [0.01, 100.0]$: Gradient step-size gain.
  - $a \in [0.01, 100.0]$: Random exploration perturbation variance.
* **Fixed Parameters:**
  - `epsilon`: $10^{-4}$ (decay stabilization parameter).
  - `estimation_type`: `"one-point"`.
* **Simulation Length:** `num_steps = 10000`.

#### C. Continuous-Time Nash Equilibrium Seeking (`NESeeking`)
* **Reference:** Frihauf, Krstic, & Basar (2012).
* **Search Space (8 Continuous Dimensions, Player-Specific in $[0.01, 100.0]$):**
  - Integration gains: $k[0,0], k[1,0] \in [0.01, 100.0]$
  - Perturbation amplitudes: $a[0,0], a[1,0] \in [0.01, 100.0]$
  - Dither frequencies: $\omega[0,0], \omega[1,0] \in [0.01, 100.0]$
  - Washout cutoff frequencies: $\omega_h[0], \omega_h[1] \in [0.01, 100.0]$
* **Fixed Parameters:**
  - `dt`: $0.001$ (forward Euler integration timestep).
* **Simulation Length:** `num_steps = 10000`.

---

### 3. MTBO Execution Plan & Early Stopping Budgeting

Using active task-selecting MTBO ([`MetaParamOptimization`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/mtbo.py)):
* **Active Task Selection:** In each BO iteration, Swersky's two-step heuristic queries a single game $t^* \in \{0, \dots, 99\}$ and hyperparameter $x^*$, running **only 1 simulation** per BO step.
* **Initial Quasi-Random Phase:** $n_{\text{init}} = 100$ points via Latin Hypercube Sampling, systematically paired with tasks $t_i = i \pmod{100}$ so that each of the 100 games has at least 1 evaluated baseline for the initial task covariance kernel.
* **Mathematical Convergence & Budget Ceiling Configuration:**
  - `eps_ei`: $10^{-4}$ ($0.0001$), Expected Improvement convergence threshold on $\max_x \text{EI}_{\text{avg}}(x)$.
  - `ei_window`: $3$ consecutive steps sustaining $\max_x \text{EI}_{\text{avg}} < 10^{-4}$ to confirm spatial convergence.
  - `min_bo_steps`: Baseline exploration floor of $\max(20, M // 2) = 50$ steps before early stopping is activated.
  - `max_bo_steps`: Automatically set via the dimension- and task-scaling heuristic $\min(M \cdot \max(5, 2 \cdot d), 1500)$ ($500$ for `ZOGrandientPlay`, $1500$ for `NESeeking`) when set to `None`.
* **Surrogate Model & Warm-Starting:**
  - Persistent `MultiTaskGP` surrogate fitted once initially (`maxiter=30`), updated via `set_train_data(train_x, targets, strict=False)` and warm-started (`maxiter=10`) on every subsequent step.
  - Adaptive garbage collection interval $K(100) = 25$ steps.
* **One-Shot Post-BO Verification Safeguard:**
  - Surrogate-proposed game-specific candidate parameters $\hat{\theta}_j^*$ are empirically verified on game $j$. If $y_{\text{ver}, j} \le y^*_{\text{anchor}, j}$, the candidate is accepted; otherwise falls back to observed anchor $\theta^*_{\text{anchor}, j}$.
* **Artifact Directory:** `data/03_100_games_regret/`
  - `zo_100_games_results.csv`
  - `ne_100_games_results.csv`

---

### 4. Automated Selection & Optuna Baseline on 5 Diverse Representative Games

#### A. Geometric Selection of 5 Representative Games
Because plotting 100 games produces visual clutter, 5 representative games are selected programmatically from the 100-game ensemble using objective geometric and algebraic criteria:
1. **Game 1 (Extreme Ill-Conditioned):** Maximizes $\kappa(H_{\text{sym}}) = \lambda_{\max} / \lambda_{\min}$.
2. **Game 2 (Well-Behaved / Isotropic):** Minimizes condition number $\kappa(H_{\text{sym}}) \approx 1.0$.
3. **Game 3 (Narrow Action Space):** Minimizes bounding box volume $\prod (ub - lb)$.
4. **Game 4 (Highly Asymmetric Coupling):** Maximizes anti-symmetric norm $\|H - H^T\|_F$.
5. **Game 5 (Boundary-Adjacent Equilibrium):** Minimizes distance of true NE $u^*$ to bounds.

#### B. Per-Game Optuna Baseline Optimization
To verify both the universal average prediction and individual game-specific optima, run an independent Optuna TPE optimization on each of the 5 representative games:
* **Algorithms:** Evaluated for both `ZOGrandientPlay` (2D) and `NESeeking` (8D).
* **Search Space:** Identical $[0.01, 100.0]$ bounds.
* **Objective:** Direct single-game simulation minimizing $\log_{10}(\text{Total Regret})$ over 10,000 steps.
* **Stopping Criteria:** Optuna early stopping callback with `patience = 25`, `tau = 0.005` ($0.5\%$), and `max_trials = 150`.
* **Output:** Saves `optuna_rep_games_summary.csv` in `data/03_100_games_regret/`.

---

### 5. Action Trajectory Visualization Architecture

For each of the 5 representative games, generate a 2x2 subplot trajectory comparison over 10,000 steps:
* **Top-Left (Row 1, Col 1):** Player 1 Action $u_1(t)$ under **MTBO Game-Specific Optimum $\theta_j^*$**.
* **Top-Right (Row 1, Col 2):** Player 1 Action $u_1(t)$ under **MTBO Universal Default Optimum $\theta_{\text{default}}^*$**.
* **Bottom-Left (Row 2, Col 1):** Player 2 Action $u_2(t)$ under **MTBO Game-Specific Optimum $\theta_j^*$**.
* **Bottom-Right (Row 2, Col 2):** Player 2 Action $u_2(t)$ under **MTBO Universal Default Optimum $\theta_{\text{default}}^*$**.
*(Overlaid with Optuna best trajectory and dashed black line for true NE $u^*$).*

---

### 6. Quantitative Analysis & Reporting

At the conclusion of the notebook:
1. **Representative Games Comparison Table:**
   - Display a Pandas DataFrame comparing achieved total regret across:
     - Optuna Tailored Optima (direct ground truth baseline)
     - MTBO Tailored Optima $\theta_j^*$
     - MTBO Universal Default $\theta_{\text{default}}^*$
2. **Ensemble Statistics across All 100 Games:**
   - Evaluate `evaluate_task` (10,000 steps) for both tailored and default parameters across all 100 games.
   - Report mean, median, standard deviation, and maximum regret.
   - Quantify the empirical **Tunability Index**:
     $$\Delta \text{Loss}_j = \text{Loss}_j(\theta_{\text{default}}^*) - \text{Loss}_j(\theta_j^*)$$
3. **Training Time Benchmarking & 100-Game Optuna Extrapolation:**
   - Record exact wall-clock training durations:
     - MTBO total execution time ($T_{\text{MTBO}}$) for both `ZOGrandientPlay` and `NESeeking`.
     - Optuna total execution time across the 5 representative games ($T_{\text{Optuna, 5}}$).
   - Extrapolate the total computational time to optimize all 100 games individually with Optuna:
     $$T_{\text{Optuna, 100}}^{\text{extrap}} = \frac{100}{5} \times T_{\text{Optuna, 5}} = 20 \times T_{\text{Optuna, 5}}$$
   - Compute computational speedup/efficiency ratio:
     $$\text{Speedup Factor} = \frac{T_{\text{Optuna, 100}}^{\text{extrap}}}{T_{\text{MTBO}}}$$
   - Compare total simulation queries required: MTBO active task sampling vs 100 independent Optuna studies.
4. **Automated Morning Markdown Report (`100_games_comparative_report.md`):**
   - Automatically compile and write a structured executive report to `data/03_100_games_regret/100_games_comparative_report.md`.
   - Sections:
     - **Executive Summary:** Overview of results and key takeaways.
     - **Computational Efficiency & Timing Benchmark:** MTBO runtime vs. 100-game Optuna extrapolation, speedup ratios, and query budgets.
     - **Representative Games Analysis:** Ground-truth Optuna loss vs. MTBO Tailored vs. MTBO Universal Default.
     - **100-Game Ensemble Statistics & Tunability:** Mean, median, std, max, and Tunability Index distributions.
     - **Strategic Recommendation:** Rigorous evaluation of whether to retain MTBO or adopt individual Optuna tuning for final thesis benchmarks.
