# Implementation Specification 05: Multi-Acquisition Multi-Task Bayesian Optimization & High-Dimensional Benchmark

**Status:** Draft Approved  
**Target Module:** `tests/benchmark.py` (Experimental Testbed) & Future `payoffnash/mtbo.py` Integration  
**Test Algorithm:** `NESeeking` with full parameter variability ($d = 8$, fixed $dt$)  
**Ensemble Scales:** $M \in \{1, 5, 10, 20\}$ strongly monotone quadratic games  

---

### 1. Context, Mathematical Notation & Complexity Definitions

#### A. Definitions of Variables
To ensure strict mathematical clarity regarding algorithmic complexity and data tensor dimensions:
* **$M$ (`num_tasks`):** The number of games (tasks) in the ensemble.
* **$d$ (`dim`):** The dimensionality of the continuous hyperparameter space. For `NESeeking` on $N=2, m=1$ games with all parameters free except $dt$, $d = 8$.
* **$K$ (`num_bo_steps`):** The sequential Bayesian Optimization step index ($k \in \{1, \dots, K\}$).
* **$N_{\text{init}}$:** Number of initial observations collected during the quasi-random Sobol/LHS initialization phase.
* **$N_{\text{total}}$:** The total cumulative number of evaluated $(x, y)$ observations stored in the surrogate GP training dataset across all tasks:
  $$N_{\text{total}} = \sum_{j=1}^M N_j$$
  where $N_j$ is the number of evaluations performed on task $j$.

#### B. Asymptotic Complexity in Terms of $M, K, N_{\text{total}}, d$
Because observations are unaligned (different tasks evaluated at different continuous hyperparameter candidates $\theta$), BoTorch's generic `MultiTaskGP` models cross-task covariance using the Intrinsic Coregionalization Model (ICM) kernel:
$$K((x, t), (x', t')) = k_X(x, x') \cdot K_T(t, t')$$
* **Training Data Shape:** `train_x` has shape $(N_{\text{total}}, d + 1)$ (where column $d$ is the task index $t \in \{0, \dots, M-1\}$), and `train_y` has shape $(N_{\text{total}}, 1)$.
* **Surrogate Fitting Complexity:** Cholesky factorization and determinant calculation over the full kernel matrix scale as:
  $$\mathcal{O}(N_{\text{total}}^3) = \mathcal{O}\left(\left(N_{\text{init}} + \sum_{k=1}^K E_k\right)^3\right)$$
  where $E_k$ is the number of evaluations conducted in step $k$.
  - In **Approach A ($2M$ with Active Pruning)**: $E_k = M + |\mathcal{A}_k|$, so $N_{\text{total}} \le N_{\text{init}} + 2KM$.
  - In **Approach B ($M+1$ with Active Pruning)**: $E_k = 1 + |\mathcal{A}_k|$, so $N_{\text{total}} \le N_{\text{init}} + K(M + 1)$.
* **Acquisition Optimization Complexity:**
  At each step $k$, optimizing $1 + |\mathcal{A}_k|$ acquisition functions via L-BFGS-B with $R$ restarts and $S$ raw samples scales as:
  $$\mathcal{O}\left((1 + |\mathcal{A}_k|) \cdot R \cdot I_{\text{max}} \cdot (d \cdot N_{\text{total}})\right)$$
  where $I_{\text{max}}$ is the maximum L-BFGS-B iterations.

---

### 2. High-Dimensional Parameter Space Formulation (`NESeeking`, $d = 8$)

The benchmark evaluates the Nash Equilibrium Seeking algorithm (`NESeeking`) with full player-specific variability across all hyperparameter matrices on 2-player games ($N = 2, m = 1$):
* **Gain Matrix $k \in \mathbb{R}^{2 \times 1}$:** 
  `k[0,0]`, `k[1,0]` $\in [0.01, 10.0]$ (logarithmic search scale)
* **Dither Amplitude $a \in \mathbb{R}^{2 \times 1}$:** 
  `a[0,0]`, `a[1,0]` $\in [0.01, 5.0]$ (logarithmic search scale)
* **Dither Frequency $\omega \in \mathbb{R}^{2 \times 1}$:** 
  `omega[0,0]`, `omega[1,0]` $\in [10.0, 50.0]$ (logarithmic search scale)
* **High-Pass Filter Frequency $\omega_h \in \mathbb{R}^{2}$:** 
  `omega_h[0]`, `omega_h[1]` $\in [1.0, 20.0]$ (logarithmic search scale)
* **Time Step $dt$:** Fixed constant $dt = 0.01$.

Total continuous hyperparameter dimension: $d = 8$.

---

### 3. Multi-Acquisition Architecture & Objectives

Instead of scalarizing the ensemble with a single average acquisition function (which leaves weakly correlated or anti-correlated tasks un-explored), the surrogate model maintains $M + 1$ distinct acquisition objectives at every active iteration:

#### A. Objectives Definition
Using BoTorch's `LinearMCObjective`:
1. **Ensemble Average Objective ($\mathcal{O}_0$):**
   $$w_{\text{avg}} = \left[-\frac{1}{M}, -\frac{1}{M}, \dots, -\frac{1}{M}\right]^T \in \mathbb{R}^M, \quad \mathcal{O}_0(Y) = -\frac{1}{M}\sum_{j=1}^M Y_j$$
2. **Task-Specific Objectives ($\mathcal{O}_j$ for $j \in \{1, \dots, M\}$):**
   $$w_j = [0, \dots, -1, \dots, 0]^T \in \mathbb{R}^M \quad (\text{with } -1 \text{ at index } j-1), \quad \mathcal{O}_j(Y) = -Y_j$$

#### B. Acquisition Functions
For each objective $j \in \{0, 1, \dots, M\}$, instantiate BoTorch's log-expected improvement:
$$\alpha_j(\theta) = \text{qLogExpectedImprovement}\left(\text{model}, \text{best\_f}_j, \text{sampler}, \mathcal{O}_j\right)$$
where $\text{best\_f}_j = \max_{i} \mathcal{O}_j(Y_i)$.

Continuous optimization over the unit cube $[0, 1]^d$ is conducted via BoTorch's `optimize_acqf`:
$$\theta_j^*, \text{acq\_val}_j = \text{optimize\_acqf}(\alpha_j, \text{bounds}=[0, 1]^d, q=1, \text{restarts}=5, \text{raw\_samples}=64)$$
$$\text{EI}_j^* = \exp(\text{acq\_val}_j)$$

---

### 4. Active Task Pruning & Dual Early Stopping

#### A. Dynamic Active Set ($\mathcal{A}_k$)
To prevent wasting computational cycles and simulation evaluations on tasks whose hyperparameter basins have already converged:
1. Define the active set at iteration $k$:
   $$\mathcal{A}_k = \{j \in \{1, \dots, M\} \mid \text{EI}_j^* \ge \varepsilon_{\text{EI}}\}$$
   where $\varepsilon_{\text{EI}} = 10^{-4}$.
2. Tasks with $\text{EI}_j^* < 10^{-4}$ are pruned from individual query generation.
3. As optimization progresses, $|\mathcal{A}_k|$ monotonically decreases, accelerating the remaining BO iterations.

#### B. Dual Early Stopping Condition
Optimization halts early if and only if **both**:
1. The ensemble average acquisition improvement has converged: $\text{EI}_{\text{avg}}^* < \varepsilon_{\text{EI}}$ across `ei_window = 3` consecutive iterations.
2. **AND** the active task set is empty: $|\mathcal{A}_k| = 0$ (all $M$ individual tasks have achieved expected improvement $< \varepsilon_{\text{EI}}$).

---

### 5. Evaluation Strategies: Approach A vs. Approach B

The experimental testbed `tests/benchmark.py` directly compares:

#### Approach A ($2M$ Evaluations with Pruning)
In each BO step $k$:
1. Propose $\theta_{\text{avg}}^*$ via $\alpha_0$.
2. Simulate **all $M$ games** at $\theta_{\text{avg}}^*$ ($M$ evaluations).
3. Propose $\theta_j^*$ via $\alpha_j$ for each active task $j \in \mathcal{A}_k$.
4. Simulate **game $j$** at $\theta_j^*$ ($|\mathcal{A}_k|$ evaluations).
* **Total step simulations:** $M + |\mathcal{A}_k| \le 2M$.

#### Approach B ($M+1$ Evaluations with Pruning)
In each BO step $k$:
1. Propose $\theta_{\text{avg}}^*$ via $\alpha_0$.
2. **Active Task Selection:** Identify task $t^* \in \{1, \dots, M\}$ yielding maximal single-task expected improvement at $\theta_{\text{avg}}^*$:
   $$t^* = \arg\max_{t \in \{1, \dots, M\}} \text{EI}_t(\theta_{\text{avg}}^*)$$
   Simulate **only task $t^*$** at $\theta_{\text{avg}}^*$ ($1$ evaluation).
3. Propose $\theta_j^*$ via $\alpha_j$ for each active task $j \in \mathcal{A}_k$.
4. Simulate **game $j$** at $\theta_j^*$ ($|\mathcal{A}_k|$ evaluations).
* **Total step simulations:** $1 + |\mathcal{A}_k| \le M + 1$.

---

### 6. Tunability Anomaly Detection & Safeguard

Because evaluations are unaligned, a default parameter $\theta_{\text{def}}^*$ could theoretically yield a lower loss on game $j$ than game $j$'s tailored candidate $\theta_j^*$ ($y_j(\theta_{\text{def}}^*) < y_j^*$).

#### A. Runtime Warning
Whenever this condition is detected during post-optimization extraction, issue an explicit warning:
```python
if y_def_j < y_tailored_j:
    warnings.warn(
        f"[MTBO Anomaly] Game {j}: Universal default theta outperformed tailored theta "
        f"(y_def={y_def_j:.5f} < y_tailored={y_tailored_j:.5f}). "
        f"Applying non-negative tunability safeguard.",
        UserWarning,
    )
```

#### B. Empirical Safeguard
To guarantee non-negative tunability $d_j \ge 0$ without surrogate extrapolation:
$$\theta_j^* \leftarrow \arg\min_{\theta \in \{\theta_j^*, \theta_{\text{def}}^*\}} y_j(\theta), \quad y_j^* \leftarrow \min(y_j^*, y_j(\theta_{\text{def}}^*))$$

---

### 7. Fair Optuna Baseline Specification

To ensure neither advantage nor disadvantage:
* **Search Space:** Exact same bounds across all $d = 8$ parameters using `suggest_float(..., log=True)`.
* **Objective:** Exact same `quadratic_game_regret_cost` with fixed $dt = 0.01$ and 5000 simulation steps.
* **Per-Game Independent Studies:** Optuna runs an independent TPE study for each game $j \in \{1, \dots, M\}$.
* **Stopping Condition:** Early stopping callback with patience $W = 15$ trials when relative improvement is $< 10^{-4}$.
* **Default Extraction from Optuna:** 
  To extract Optuna's universal default $\theta_{\text{def}}^{\text{Optuna}}$, evaluate all explored configurations across all games and select the point minimizing the ensemble average, or evaluate the 1-game default baseline directly.

---

### 8. Benchmark Matrix & Deliverables

#### A. Experimental Matrix
The script `tests/benchmark.py` will execute the head-to-head comparison across:
1. $M = 1$ Game (Sanity check & Single-task equivalence)
2. $M = 5$ Games (Small ensemble)
3. $M = 10$ Games (Standard thesis ensemble)
4. $M = 20$ Games (Stress-test for scaling)

#### B. Metrics Tracked
1. **Solution Quality:**
   - Ground Truth Ensemble Average Loss of $\theta_{\text{def}}^*$ across all $M$ games.
   - Mean Tailored Loss $\frac{1}{M}\sum_{j=1}^M y_j^*$ across all $M$ games.
   - Tunability Index distribution: $d_j = y_j(\theta_{\text{def}}^*) - y_j^*$.
2. **Correlation Learning Fidelity:**
   - Frobenius norm error $\|K_T - C_{\text{true}}\|_F$ between GP learned task covariance and empirical cross-task Pearson correlation.
3. **Computational Efficiency:**
   - Total wall-clock time (seconds).
   - Total number of simulation evaluations performed.
   - GP fitting time vs. Acquisition optimization time breakdown.
