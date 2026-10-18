# Implementation Specification 02: Multi-Acquisition Multi-Task Bayesian Optimization for Metaparameter Tuning

**Context:** This module implements the Bayesian optimization engine required to compute the Tunability Index across an ensemble of strongly monotone quadratic games. It implements BoTorch's generic Multi-Task Gaussian Process (`MultiTaskGP`) using an Intrinsic Coregionalization Model (ICM) kernel with vectorized closed-form analytic Expected Improvement across candidate pools. Hyperparameter candidates are optimized on a logarithmic scale across $M+1$ distinct objectives (one ensemble average objective and $M$ task-specific objectives) with independent per-task pruning (via individual EI threshold or incumbent stagnation), multi-core parallel task simulation, memory hygiene, post-evaluation adaptive refinement for anomalies, and an analytical Leave-One-Out (LOO) validation engine.

---

### 1. Module Objectives
Implement the modernized class `MetaParamOptimization` managing the BoTorch-based MTBO workflow. The class:
1. Evaluates a unified zeroth-order learning estimator (`ZOGrandientPlay` or `NESeeking`) across an ensemble of $M$ strongly monotone quadratic games ($M \ge 1$, benchmarked on $M \in \{1, 5, 10, 20, 100\}$).
2. Maps all continuous multiplicative hyperparameters (learning rate $v$, gain $k$, dither amplitude $a$, frequency $\omega$, etc.) to a **logarithmic search space**, ensuring equal probability mass and stationary GP kernel resolution across orders of magnitude.
3. Fits a multi-task GP surrogate (`MultiTaskGP` with task index feature $t \in \{0, \dots, M-1\}$) over unaligned observation tuples $(x, t, y)$, capturing cross-task covariance ($K = k_X \otimes K_T$) while permitting task-specific queries.
4. Uses a uniform, scalable **vectorized analytic Expected Improvement** acquisition policy across quasi-random candidate pools for all ensemble sizes $M$, completely eliminating unscalable sequential Monte Carlo loops (`qLogExpectedImprovement`) and artificial $M < 10$ branches.
5. Implements **parallelized task simulation** across multiple CPU cores (`n_jobs`) using process-level parallel execution to accelerate ground-truth simulations.
6. Implements **individual per-task pruning and simplified convergence**: each active task prunes independently when either its candidate EI drops below $\varepsilon_{\text{EI}}$ OR its incumbent best observed loss stagnates over a patience window. Optimization halts when all tasks have pruned or `max_bo_steps` ceiling is hit.
7. Implements **Post-Evaluation Adaptive Refinement**: Whenever an anomaly occurs ($y_j(\theta_{\text{def}}^*) < y_j^*$), triggers a dedicated refinement burst for Game $j$ initialized around $\theta_{\text{def}}^*$, guaranteeing non-negative tunability ($d_j \ge 0$).
8. Maintains strict **memory management and garbage collection** (`_garbage_collect`) between surrogate fits to prevent RAM leakage during long runs.
9. Provides a dedicated **`validate()`** method computing analytical Leave-One-Out Cross-Validation (LOO-CV) RMSE, standardized residual calibration, and Bayesian suboptimality bounds at zero extra simulation cost.

---

### 2. Architecture & The `NEEstimator` Interface

To ensure strict type safety and architectural consistency, learning algorithms inherit from the abstract base class `NEEstimator`:

1. **Abstract Base Class `NEEstimator` (in `payoffnash/common.py`):**
   - Requires `estimate` method:
     ```python
     @abstractmethod
     def estimate(
         self, initial_actions: np.ndarray, num_steps: int, max_rel_error: float, time_limit: float
     ) -> EstimateHistory:
         """Executes seeking dynamics and returns EstimateHistory."""
     ```
   - Requires parameter introspection method:
     ```python
     @classmethod
     @abstractmethod
     def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
         """Returns dictionary mapping constructor argument names to expected array dimensions."""
     ```

2. **Pickling Guarantees for Parallel Execution:**
   - To support multiprocessing across cores, `QuadraticGame` and its payoff callables must be fully serializable via Python's standard `pickle`.
   - Payoff callables are encapsulated in top-level callable classes (`QuadraticPayoffFunction`) rather than local closures.

3. **Broadcasting and Indexed Parameter Support:**
   - Supports both collective parameter broadcasting (e.g., `"k": (0.1, 5.0)`) and player-specific indexed parameters (e.g., `"k[0,0]": (0.1, 10.0)`, `"omega[1,0]": (15.0, 35.0)`).
   - Modernized interface parameters: `num_steps`, `max_rel_error`, `time_limit`, `cost_function`, and `initial_actions`.

---

### 3. Hyperparameter Space, Logarithmic Mapping & Helper Functions

#### A. Logarithmic Parameter Mapping
For positive hyperparameters varying across orders of magnitude (e.g. $[0.01, 100.0]$):
$$u_i = \log_{10}(\theta_i) \in [\log_{10}(L_i), \log_{10}(U_i)]$$
$$x_{\text{norm}, i} = \frac{\log_{10}(\theta_i) - \log_{10}(L_i)}{\log_{10}(U_i) - \log_{10}(L_i)} \in [0, 1]$$
$$\theta_i = 10^{\log_{10}(L_i) + x_{\text{norm}, i} \cdot (\log_{10}(U_i) - \log_{10}(L_i))}$$

#### B. Core Class Helper Methods
To keep the optimization loop clean, modular, and self-documenting:
1. `_normalize(theta: np.ndarray) -> np.ndarray`: Maps physical parameters $\theta \in \mathbb{R}^d$ to unit hypercube $[0, 1]^d$ using logarithmic mapping for positive domains.
2. `_unnormalize(u: np.ndarray) -> np.ndarray`: Inverts unit hypercube $u \in [0, 1]^d$ back to original physical parameter scale $\theta \in \mathbb{R}^d$.
3. `_build_estimator_kwargs(theta: np.ndarray, game: QuadraticGame) -> dict[str, Any]`: Introspects `algorithm_class.get_param_specs(game)` and constructs correctly-shaped multidimensional arrays from flat parameter vectors, merging `fixed_params`.
4. `_flatten_hyperparams(theta: np.ndarray, game: QuadraticGame) -> dict[str, Any]`: Converts multidimensional algorithm parameters into flat string keys matching `search_space` format for clean CSV serialization.
5. `_compute_analytic_ei(delta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor`: Computes exact closed-form Expected Improvement $\text{EI} = \Delta \Phi(\Delta / \sigma) + \sigma \phi(\Delta / \sigma)$ using standard normal CDF and PDF in a single vectorized PyTorch operation.
6. `_garbage_collect() -> None`: Manages memory by clearing PyTorch CUDA caches (if present) and invoking Python `gc.collect()`.

#### C. Scale-Invariant Parameter Scaling (`scaled_params`)
To ensure coordinate invariance across game ensembles with heterogeneous action spaces (where bounding box diameters $W_i = \text{ub}_i - \text{lb}_i$ can span orders of magnitude), `MetaParamOptimization` accepts an optional `scaled_params: Optional[dict[str, str]] = None` configuration:
- `"min_box_width"`: Scales the parameter by the bottleneck dimension across all players $W_{\min} = \min_{i} (\text{ub}_i - \text{lb}_i)$.
  For `ZOGrandientPlay`: `scaled_params = {"a": "min_box_width"}`.
  The hyperparameter $\alpha \in [0.005, 2.0]$ represents relative exploration ($0.5\%$ to $200\%$ of minimum width). Effective scalar amplitude passed to the estimator:
  $$a_{\text{eff}} = \alpha \cdot \min_{i} (\text{ub}_i - \text{lb}_i)$$
- `"player_box_width"` or `"box_width"`: Scales player-specific parameters $a[i,0]$ by player $i$'s individual bounding box width $W_i = \text{ub}_i - \text{lb}_i$.
  For `NESeeking`: `scaled_params = {"a[0,0]": "player_box_width", "a[1,0]": "player_box_width"}` (or `"a": "player_box_width"`).
  Effective amplitude:
  $$a_i = \alpha_i \cdot (\text{ub}_i - \text{lb}_i)$$
When constructing estimator arguments in `_build_estimator_kwargs(theta, game)`, MTBO automatically scales the corresponding entries by the game's actual action boundaries. When `scaled_params=None` (the default), MTBO operates in unscaled physical units for full backward compatibility.

---

### 4. Surrogate Model Formulation: `MultiTaskGP`

To natively handle unaligned task queries without enforcing an impractical full grid of evaluations:
* **Input Tensor:** `train_x` of shape $(N_{\text{total}}, d + 1)$, where column $d$ is the task index $t \in \{0, \dots, M-1\}$.
* **Target Tensor:** `train_y` of shape $(N_{\text{total}}, 1)$, standardized via `Standardize(m=1)`.
* **Covariance Structure:**
  $$K((x, t), (x', t')) = k_X(x, x') \cdot K_T(t, t')$$
  where $k_X$ is a Matérn 5/2 kernel over hyperparameter space, and $K_T$ is a low-rank coregionalization matrix $W W^T + \text{diag}(v)$ (`PositiveIndexKernel`).
* **Fitting:** Re-instantiate `MultiTaskGP(train_x, train_y, task_feature=-1)` with warm-started state dict and fit MLL via `fit_gpytorch_mll` with `maxiter=25`.

---

### 5. Unified Vectorized Acquisition Policy (Approach B Architecture)

Completely eliminates `qLogExpectedImprovement` loops and arbitrary branches based on ensemble size $M$.

#### A. Dimension-Scaled Candidate Pool
At each BO step $k$:
1. Draw a scrambled quasi-random candidate pool from the unit hypercube:
   $$X_{\text{cand}} \sim \text{Sobol}(d, N_{\text{cand}}), \quad N_{\text{cand}} = \min(1024, \max(256, 128 \cdot d))$$
2. Compute the GP posterior distribution across the candidate pool in a single forward pass:
   $$\boldsymbol{\mu}, \boldsymbol{\sigma} = \text{posterior}(X_{\text{cand}}) \in \mathbb{R}^{N_{\text{cand}} \times M}$$

#### B. Exact Vectorized Expected Improvement
1. **Universal Ensemble Candidate ($\theta_{\text{def}}^*$):**
   - Predicted ensemble mean and pooled variance:
     $$\bar{\mu}(x) = \frac{1}{M}\sum_{j=1}^M \mu_j(x), \quad \bar{\sigma}^2(x) = \frac{1}{M^2}\sum_{j=1}^M \sigma_j^2(x)$$
   - Candidate $\theta_{\text{def}}^*$ maximizes analytic EI on the ensemble mean:
     $$\theta_{\text{def}}^* = \arg\max_{x \in X_{\text{cand}}} \text{EI}\left(\bar{y}^* - \bar{\mu}(x), \bar{\sigma}(x)\right)$$
   - Evaluated on the single task $t^*$ maximizing single-task EI at $\theta_{\text{def}}^*$:
     $$t^* = \arg\max_{t \in \{0, \dots, M-1\}} \text{EI}_t(\theta_{\text{def}}^*)$$
2. **Game-Specific Tailored Candidates ($\theta_j^*$):**
   - For all active tasks $j \in \mathcal{A}_k$ simultaneously in a single tensor operation:
     $$\text{EI}_j(X_{\text{cand}}) = (y_j^* - \boldsymbol{\mu}[:, j])\Phi(z_j) + \boldsymbol{\sigma}[:, j]\phi(z_j), \quad z_j = \frac{y_j^* - \boldsymbol{\mu}[:, j]}{\boldsymbol{\sigma}[:, j]}$$
   - Tailored candidate for task $j$:
     $$\theta_j^* = X_{\text{cand}}\left[\arg\max_{i} \text{EI}_j(X_{\text{cand}}[i])\right], \quad \text{EI}_j^* = \max_i \text{EI}_j(X_{\text{cand}}[i])$$

---

### 6. Individual Per-Task & Universal Candidate Pruning & Halting Condition

#### A. Individual Task Pruning Criteria
At step $k$, each active task $j \in \mathcal{A}_k$ is checked independently. Task $j$ is **pruned** ($\mathcal{A}_{k+1} = \mathcal{A}_k \setminus \{j\}$) if **either** of the following conditions is satisfied:
1. **Acquisition Exhaustion:**
   $$\text{EI}_j^* < \varepsilon_{\text{EI}} \quad (\text{default } 10^{-4})$$
2. **Incumbent Stagnation:**
   Over the last $W$ iterations ($W = \text{patience}$, default 6):
   $$\Delta y_j^{(W)} = y_j^{*(k - W)} - y_j^{*(k)} < \varepsilon_{\text{tol}} \quad (\text{default } 10^{-4})$$
   (Only evaluated once history length $\ge W$).

#### B. Universal Default Candidate Stopping Criteria
The universal default candidate $\theta_{\text{def}}^*$ is treated as an active pursuit with its own state `def_active: bool = True`. It is pruned (`def_active = False`, halting further $t^*$ queries) if **either**:
1. **Acquisition Exhaustion on Ensemble Average:**
   $$\text{EI}_{\text{avg}}^* < \varepsilon_{\text{EI}} \quad (\text{default } 10^{-4})$$
2. **Incumbent Stagnation on Ensemble Average:**
   Over the last $W$ iterations:
   $$\Delta \bar{y}^{(W)} = \bar{y}^{*(k - W)} - \bar{y}^{*(k)} < \varepsilon_{\text{tol}} \quad (\text{default } 10^{-4})$$

#### C. Algorithm Halting Condition
The optimization loop terminates immediately when:
1. **All Pursuits Pruned:**
   $$\text{not def\_active and len}(\mathcal{A}_k) == 0 \implies \text{stopped\_early} = \text{True}, \quad \text{stop\_reason} = \text{"all\_tasks\_pruned"}$$
2. **Ceiling Safeguard (`max_bo_steps`):**
   A safety ceiling (default 200) is provided. In normal operation, individual task pruning and universal candidate convergence halt the loop well before 200 steps. Reaching `max_bo_steps` sets `stop_reason = "max_bo_steps_reached"`, signaling that the stopping tolerances were unfeasible.

#### D. Execution Logging & Progress Tracking
1. **Pruning Event Notifications:** Whenever any individual task $j$ or the Universal Default candidate is pruned, an immediate notification is emitted to stdout documenting the exact reason and Expected Improvement at prune time:
   - *Expected Improvement floor:* `[Task Pruned] Game j: Expected Improvement floor (EI = ... < tol ...)`
   - *Incumbent Stagnation:* `[Task Pruned] Game j: Incumbent Stagnation (imp over W steps = ... < tol ..., EI at prune = ...)`
2. **5-Iteration Summary Table:** Every 5 BO iterations (`(step + 1) % 5 == 0`), a tabular status report across all $M+1$ tasks ($M$ individual games + Universal Default) is printed to stdout displaying `Task`, `Status` (ACTIVE / PRUNED), `Best Loss`, `Current EI`, and `Patience` remaining countdown (`rem_patience / patience`).

---

### 7. Multi-Core Parallel Task Simulation

To maximally utilize modern multi-core CPUs:
1. **Execution Model:**
   - Parameter `n_jobs: Optional[int] = None` in `MetaParamOptimization.__init__`. Defaults to `os.cpu_count()`.
   - Initial design evaluations ($N_{\text{init}} \times M$) and per-step active task simulations ($1 + |\mathcal{A}_k|$) are distributed across worker processes using `concurrent.futures.ProcessPoolExecutor`.
2. **Pickling Architecture:**
   - `QuadraticGame` objects use picklable top-level payoff structures (`QuadraticPayoffFunction`), eliminating closure serialization errors.
3. **Chunking & Thread Safety:**
   - Worker processes execute in isolated environments with PyTorch single-threading (`torch.set_num_threads(1)`) to avoid BLAS/OpenMP contention.

---

### 8. Post-Evaluation Surrogate Posterior Continuous Polish & Anomaly Detection

To bridge the discretization gap in higher dimensions at minimal computational cost, MTBO executes a targeted continuous polish (Option B) across all tasks simultaneously before returning results:

#### A. Surrogate Posterior Continuous Polish (L-BFGS-B on $\mu(x)$)
1. **Model Access:** Uses `self.last_model` (`MultiTaskGP`), which already captures the complete cross-task covariance dataset.
2. **Local Trust Region:** For each task $j$, initialize at unit seed $u_j^* = \text{\_normalize}(\theta_j^*)$ with local trust bounds:
   $$B_j = [\max(0, u_j^* - \delta), \min(1, u_j^* + \delta)]^d, \quad \delta = 0.10 \text{ (10% radius)}$$
3. **Autograd Continuous Minimization:** Using PyTorch autograd gradients with L-BFGS-B, continuously minimize the posterior mean:
   $$x_j^{\text{cont}} = \arg\min_{u \in B_j} \mu_j(u)$$
   Converges in $< 0.02$s per task without extra game simulations.
4. **Universal Default Polish:** Continuously minimize the predicted ensemble average mean:
   $$x_{\text{def}}^{\text{cont}} = \arg\min_{u \in B_{\text{def}}} \bar{\mu}(u) = \arg\min_{u \in B_{\text{def}}} \frac{1}{M}\sum_{j=0}^{M-1} \mu_j(u)$$

#### B. Single Parallel Verification Batch
Dispatches a single parallel simulation batch across worker processes:
1. Simulates all $M$ games at their continuous candidates $x_j^{\text{cont}} \to y_j^{\text{cont}}$.
2. Simulates all $M$ games at the continuous default candidate $x_{\text{def}}^{\text{cont}} \to y_{j, \text{def}}^{\text{cont}}$.
Total time on an 8-core CPU is $\approx 20\text{ seconds}$ for 100 games.

#### C. Three-Way Optima Comparison & Anomaly Warning
For each task $j \in \{0, \dots, M-1\}$, the algorithm compares losses across three distinct sources:
1. **Continuous Polish:** $y_j^{\text{cont}}$ from continuous L-BFGS-B on $\mu_j(x)$.
2. **Discrete Tailored BO:** $y_j^{\text{bo}}$ from the best historical evaluation during the discrete BO loop.
3. **Universal Default:** $y_{j, \text{def}}^{\text{cont}}$ from the polished universal default.

**Selection & Anomaly Handling:**
* Assign $\theta_j^*$ to the candidate yielding $\min(y_j^{\text{cont}}, y_j^{\text{bo}}, y_{j, \text{def}}^{\text{cont}})$.
* **Anomaly Warning:** If the universal default is the strictly lowest loss ($y_{j, \text{def}}^{\text{cont}} < \min(y_j^{\text{cont}}, y_j^{\text{bo}})$):
  ```python
  warnings.warn(
      f"[MTBO Anomaly] Game {j}: Universal default outperformed both continuous and discrete tailored optima "
      f"(y_def={y_def_j:.5f} < min(y_cont={y_cont_j:.5f}, y_bo={y_bo_j:.5f})).",
      UserWarning,
  )
  ```

#### D. Summary Print Statement
A clear executive summary is printed to stdout declaring the winner and task breakdown:
```text
=== MTBO Post-Evaluation Polish Summary ===
Total Tasks: M
1. Continuous exploration:        N_cont (XX.X%) | Avg Loss: 0.XXXXX
2. Individual task (Discrete BO): N_bo   (XX.X%) | Avg Loss: 0.XXXXX
3. Default optima:                N_def  (XX.X%) | Avg Loss: 0.XXXXX
-------------------------------------------
BEST OPTIMA AFTER POLISH: [CONTINUOUS EXPLORATION | INDIVIDUAL TASK | DEFAULT OPTIMA]
===========================================
```

---

### 9. Analytical Leave-One-Out Validation Engine (`validate()`)

Computes closed-form LOO cross-validation without running additional game simulations:
$$\mu_{-i}(x_i) = y_i - \frac{[\mathbf{K}^{-1} \mathbf{y}]_i}{[\mathbf{K}^{-1}]_{ii}}, \qquad \sigma_{-i}^2(x_i) = \frac{1}{[\mathbf{K}^{-1}]_{ii}}$$

Returns:
* `loo_rmse`: Analytical root mean squared error.
* `loo_smse`: Standardized mean squared error ($\text{SMSE} < 1.0$).
* `calibration_std`: Standard deviation of standardized residuals $z_i$ ($s_z \approx 1.0$).
* `suboptimality_bounds`: 95% Bayesian credible suboptimality upper bounds across tasks.
* `anomalies_detected`: Count of anomalies prior to refinement.
* `refinements_executed`: Count of refinement bursts executed.

---

### 10. Class Signature & Parameter Specifications

```python
class MetaParamOptimization:
    def __init__(
        self,
        game_ensemble: list[QuadraticGame],
        search_space: dict[str, tuple[float, float]],
        algorithm_class: type[NEEstimator],
        output_path: Optional[str] = None,
        cost_function: Optional[Callable[[QuadraticGame, EstimateHistory], float]] = None,
        num_steps: int = 10000,
        max_rel_error: float = 0.0,
        time_limit: float = np.inf,
        n_init: Optional[int] = 10,
        max_bo_steps: int = 200,
        eps_ei: float = 1e-4,
        patience: int = 6,
        tol_stagnation: float = 1e-4,
        n_jobs: Optional[int] = None,
        seed: Optional[int] = None,
        fixed_params: Optional[dict[str, Any]] = None,
        scaled_params: Optional[dict[str, str]] = None,
        initial_actions: Optional[np.ndarray] = None,
    ):
```

---

### 11. Validation Test Plan (`tests/test_mtbo.py`)

1. **Introspection & Broadcasting Test:** Verify that parameter bounds broadcast correctly and indexed parameters are parsed properly.
2. **Log-Scale Mapping Test:** Verify bijective round-trip transformations between physical parameters and unit cube $[0, 1]^d$.
3. **Pickling & Parallel Simulation Test:** Verify that `QuadraticGame` instances serialize cleanly and simulate correctly across multiple parallel worker processes.
4. **Vectorized Closed-Form EI Test:** Verify that `_compute_analytic_ei` matches standard SciPy Gaussian EI.
5. **Individual Pruning & Stopping Test:** Verify that individual tasks prune when stagnated or below $\varepsilon_{\text{EI}}$, and loop halts when $\mathcal{A} = \emptyset$.
6. **Adaptive Refinement Test:** Mock an anomaly and verify local refinement guarantees $d_j \ge 0$.
7. **Surrogate `validate()` Test:** Verify that analytical LOO metrics (`loo_rmse`, `loo_smse`, `calibration_std`) are finite and correctly formatted.
8. **Scale-Invariant Parameter Scaling Test (`test_scaled_params`):** Verify that specifying `scaled_params={"a": "min_box_width"}` scales scalar $a$ by $\min_i (ub_i - lb_i)$, and `scaled_params={"a[0,0]": "player_box_width", "a[1,0]": "player_box_width"}` scales player-specific $a_i$ by $(ub_i - lb_i)$, while leaving unscaled parameters unchanged.