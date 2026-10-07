# Implementation Specification: Multi-Task Bayesian Optimization for Metaparameter Tuning

**Context:** This module implements the Bayesian optimization loop required to compute the Tunability Index across an ensemble of strongly monotone quadratic games. It implements active task-selecting Multi-Task Bayesian Optimization (MTBO) based on Kevin Swersky et al. (NIPS 2013, Section 3.2) using BoTorch's `MultiTaskGP` surrogate. Rather than evaluating all games in the ensemble at each query (block design), MTBO dynamically selects a candidate hyperparameter and game task $(x^*, t^*)$ at each step and evaluates *only that single task*, leveraging task covariance to tune both game-specific optimal hyperparameters ($\theta_j^*$) and the universal default hyperparameters ($\theta_{\text{default}}^*$) across an ensemble of 100 games.

---

### 1. Module Objectives
Implement the class `MetaParamOptimization` managing the BoTorch-based MTBO workflow. The class:
1. Evaluates a unified zeroth-order learning estimator (`ZOGrandientPlay` or `NESeeking`) across an ensemble of $M = 100$ strongly monotone quadratic games.
2. Fits an Intrinsic Coregionalization Model (ICM) Gaussian Process surrogate (`MultiTaskGP`) mapping $(x, t)$ pairs to smoothed Nash Equilibrium distances.
3. Implements the Swersky et al. (2013) two-step task-selection acquisition policy to jointly determine the next hyperparameter query $x^*$ and game $t^*$ to evaluate.
4. Extracts game-specific optima $\theta_j^*$ and universal default optima $\theta_{\text{default}}^*$, exporting results to CSV.
5. Provides automated selection and visualization of 5 diverse representative games from the 100-game ensemble, plotting actions only.

---

### 2. Architecture & The `NEEstimator` Interface

To ensure strict type safety and architectural consistency, learning algorithms must inherit from an abstract base class `NEEstimator`:

1. **Abstract Base Class `NEEstimator` (in `payoffnash/common.py`):**
   - Requires the implementation of the `estimate` method:
     ```python
     @abstractmethod
     def estimate(
         self, initial_actions: np.ndarray, **kwargs
     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
         """Executes the seeking dynamics and returns (final_state: (N, m), action_history: (T, Nm), payoff_history: (T, N))."""
     ```
   - Requires a parameter introspection method:
     ```python
     @classmethod
     @abstractmethod
     def get_param_specs(cls, game: Game) -> dict[str, tuple[int, ...]]:
         """Returns a dictionary mapping constructor argument names to their expected tensor/array dimensions."""
     ```

2. **Subclass Integration:**
   - Both `ZOGrandientPlay` and `NESeeking` inherit from `NEEstimator`.
   - `ZOGrandientPlay.get_param_specs(game)` returns scalar parameter dimensions (e.g., `{"v": (1,), "a": (1,), "epsilon": (1,)}`).
   - `NESeeking.get_param_specs(game)` returns multivariate shapes matching the full action space dimensions:
     For a game with $N$ players and $m$ action dimensions per player, parameters such as integration gains `k`, perturbation amplitudes `a`, dither frequencies `omega`, and phase shifts `phi` consist of $N$ arrays each of length $m$ (totaling $N \cdot m$ scalar parameters per type).
     `get_param_specs(game)` reports these as `{"k": (game.N, m), "a": (game.N, m), "omega": (game.N, m), "phi": (game.N, m), "dt": (1,)}`, where the total number of entries for each parameter type is $N \cdot m$.

3. **Automatic Broadcasting from Search Space:**
   - When the user defines a scalar search interval for a multivariate parameter (e.g., `"k": (0.1, 5.0)` or `"a": (0.01, 0.5)` for `NESeeking`), `MetaParamOptimization` automatically broadcasts the scalar value into a contiguous 2D `np.ndarray` of shape `(N, m)` according to `get_param_specs` before instantiating the estimator.
   - For dither frequencies `omega`, all $N \cdot m$ perturbation channels across all players are populated as a 2D array of shape `(N, m)`.

4. **Refactoring Estimators to Native Multidimensional NumPy Arrays (`(N, m)`):**
   To maximize vectorization efficiency, eliminate Python interpreter overhead, and support zero-copy PyTorch interop (clean break from `list[np.ndarray]` without legacy shims):
   - **Hyperparameters:** All multivariate parameters in `NESeeking` (`k`, `a`, `omega`, `phi`) are strictly 2D `np.ndarray` of shape `(N, m)`.
   - **Action Set Bounds:** Precompute bounding box limits `lb` and `ub` as contiguous 2D `np.ndarray` of shape `(N, m)`.
   - **Initial & Internal States:** Initial actions and internal states (`u_hat`, `state`) are strictly 2D `np.ndarray` of shape `(N, m)` instead of lists of arrays.
   - **Vectorized Inner Loop (`NESeeking`):** Compiled, vectorized array operations:
     ```python
     dist = np.minimum(u_hat - lb, ub - u_hat)
     alpha = np.clip(dist / self.a, 0.0, 1.0)
     pert = alpha * self.a * np.sin(self.omega * t + self.phi)
     u = np.clip(u_hat + pert, lb, ub)
     ```
   - **Vectorized Inner Loop (`ZOGrandientPlay`):** Draw all $N \cdot m$ perturbation samples simultaneously via `np.random.normal(state, sigma)` of shape `(N, m)` and vectorize set regularizations.
   - **Pre-Allocated Trajectories:**
     - `action_history`: `np.ndarray` of shape `(num_steps, Nm)`.
     - `payoff_history`: `np.ndarray` of shape `(num_steps, N)`.
     - Final state returned as `np.ndarray` of shape `(N, m)`.

---

### 3. Class Initialization & Inputs

`MetaParamOptimization` accepts the following parameters:

* **`game_ensemble` (`list[QuadraticGame]`):** The ensemble of strongly monotone quadratic games ($M = 100$ games, $N=2, m=1$).
* **`search_space` (`dict[str, tuple[float, float]]`):** Dictionary defining continuous lower and upper bounds for each tuned hyperparameter.
* **`algorithm_class` (`type[NEEstimator]`):** The specific estimator class being tuned (`ZOGrandientPlay` or `NESeeking`).
* **`output_path` (`Optional[str]`):** File path where final optimization results are saved in CSV format.
* **`num_iters` (`int`):** Total number of simulation steps $N_{\text{iter}}$ per game evaluation (default: `3000`).
* **`smoothing_window` (`int`):** Window size $K$ for calculating the moving-average equilibrium error (default: `50`).
* **`n_init` (`int`):** Number of initial quasi-random parameter evaluations across tasks (default: `100`, assigning 1 distinct sample per game).
* **`n_bo_steps` (`int`):** Number of sequential Bayesian optimization iterations (default: `500` to `800` for a 2–3 hour overnight run).
* **`seed` (`Optional[int]`):** Random seed for reproducible sampling and PyTorch execution.
* **`fixed_params` (`Optional[dict[str, Any]]`):** Fixed algorithm parameters (e.g., `{"estimation_type": "one-point"}` for `ZOGrandientPlay`).

---

### 4. Objective Metric & Optimization Formulation

#### A. Smoothed Error Metric ($g_j(\theta)$)
For game $j \in \{0, \dots, M-1\}$ and candidate hyperparameter vector $\theta$, evaluate the estimator for $N_{\text{iter}} = 3000$ steps. The loss metric is the average Euclidean distance to the true Nash Equilibrium $x_j^*$ over the final $K = 50$ iterations:
$$g_j(\theta) = \frac{1}{K} \sum_{t=N_{\text{iter}}-K}^{N_{\text{iter}}} \|x_t(\theta) - x_j^*\|_2$$

#### B. Alignment with BoTorch Maximization
The surrogate GP models the loss function $y = g_t(x)$ directly or as negative loss $-g_t(x)$ for maximization. With standard minimization of distance:
$$y_{\text{obs}} = g_j(\theta)$$
GP training inputs are augmented vectors $X_{\text{train}} = [\bar{\theta}, j] \in [0, 1]^D \times \{0, \dots, M-1\}$, where $\bar{\theta}$ is normalized to the unit hypercube $[0, 1]^D$.

---

### 5. Multi-Task Bayesian Optimization Loop (Swersky et al. 2013, Section 3.2)

Rather than evaluating all $M$ tasks at every step (which scales as $\mathcal{O}(M)$ simulations per BO step and is intractable for $M=100$), the algorithm employs Swersky et al.'s active task-selection procedure:

1. **Surrogate Model (`MultiTaskGP`):**
   Fit BoTorch's `MultiTaskGP(train_X, train_Y, task_feature=-1)` with outcome standardization `Standardize(m=1)`.
   The covariance function is an Intrinsic Coregionalization Model (ICM) kernel:
   $$K((x, t), (x', t')) = k_X(x, x') \cdot K_T(t, t')$$
   where $k_X$ is a Matérn 5/2 or RBF kernel over hyperparameter space, and $K_T$ is an `IndexKernel` parameterized by a low-rank task covariance matrix $W W^T + \text{diag}(v)$.

2. **Initialization:**
   - Draw $n_{\text{init}}$ initial hyperparameter points via Latin Hypercube Sampling in $[0, 1]^D$.
   - For an ensemble of $M = 100$ games, pair each initial sample with a corresponding game index $t \in \{0, \dots, M-1\}$ (1 evaluation per game for $n_{\text{init}} = 100$).
   - Run simulation for each pair and construct initial tensors:
     - `train_X`: shape $(n_{\text{init}}, D + 1)$ with column $D$ containing task index $t \in [0, M-1]$.
     - `train_Y`: shape $(n_{\text{init}}, 1)$ containing observed smoothed distances $g_t(\theta)$.

3. **Active Task-Selection Query (Two-Step Heuristic, Swersky et al. Section 3.2):**
   In each sequential iteration $k \in \{1, \dots, n_{\text{bo_steps}}\}$:
   - **Step 3a (Candidate Metaparameter $x^*$ Selection):**
     Compute the predicted ensemble mean and covariance across all $M$ tasks:
     $$\bar{\mu}(x) = \frac{1}{M} \sum_{t=0}^{M-1} \mu(x, t), \quad \bar{\sigma}^2(x) = \frac{1}{M^2} \sum_{t=0}^{M-1} \sum_{t'=0}^{M-1} \Sigma((x, t), (x, t'))$$
     Find candidate $x^* \in [0, 1]^D$ minimizing $\bar{\mu}(x)$ or maximizing Expected Improvement on the ensemble mean:
     $$\text{EI}_{\text{avg}}(x) = (\bar{y}_{\text{best}} - \bar{\mu}(x)) \Phi\left(\frac{\bar{y}_{\text{best}} - \bar{\mu}(x)}{\bar{\sigma}(x)}\right) + \bar{\sigma}(x) \phi\left(\frac{\bar{y}_{\text{best}} - \bar{\mu}(x)}{\bar{\sigma}(x)}\right)$$
     where $\bar{y}_{\text{best}} = \min_{x_i \in \mathcal{X}_{\text{eval}}} \bar{\mu}(x_i)$.
   - **Step 3b (Task $t^*$ Selection):**
     Conditioned on candidate $x^*$, select the task $t^* \in \{0, \dots, M-1\}$ yielding the highest single-task Expected Improvement:
     $$t^* = \arg\max_{t \in \{0, \dots, M-1\}} \text{EI}_t(x^*)$$
     where $\text{EI}_t(x^*) = (y_{\text{best}, t} - \mu(x^*, t))\Phi\left(\frac{y_{\text{best}, t} - \mu(x^*, t)}{\sigma(x^*, t)}\right) + \sigma(x^*, t)\phi\left(\frac{y_{\text{best}, t} - \mu(x^*, t)}{\sigma(x^*, t)}\right)$.
     In case of ties or uninformative EI, select the task with maximal posterior variance $\sigma^2(x^*, t)$ to maximally reduce uncertainty on the hyperparameter manifold.

4. **Single-Task Evaluation:**
   - Evaluate **only game $t^*$** with hyperparameter $x^*$ for $N_{\text{iter}} = 3000$ steps.
   - Record smoothed equilibrium distance $y^* = g_{t^*}(x^*)$.
   - Append $[x^*, t^*]$ to `train_X` and $y^*$ to `train_Y`.
   - Refit GP surrogate and repeat for $n_{\text{bo_steps}}$ iterations.

---

### 6. Output Specification & CSV Schema

#### A. Definition of Optima
Because task evaluations are distributed across the ensemble rather than evaluated exhaustively at every point:
* **Game-Specific Optima ($\theta_j^*$):**
  $$\theta_j^* = \arg\min_{x \in \mathcal{X}_{\text{eval}}} \mu(x, j) \quad \text{for } j \in \{0, \dots, M-1\}$$
  where $\mu(x, j)$ is the posterior GP mean for game $j$ at candidate hyperparameter $x$.
* **Universal Default Optimum ($\theta_{\text{default}}^*$):**
  $$\theta_{\text{default}}^* = \arg\min_{x \in \mathcal{X}_{\text{eval}}} \bar{\mu}(x) = \arg\min_{x \in \mathcal{X}_{\text{eval}}} \frac{1}{M}\sum_{j=0}^{M-1} \mu(x, j)$$

#### B. In-Memory Return & CSV Export
- The method `run_optimization_loop()` returns a list of dictionaries of length $M + 1$:
  - Indices `0` to `M - 1`: Game-specific optimal hyperparameter dictionaries $\theta_j^*$.
  - Index `M` (last element): The universal default hyperparameter dictionary $\theta_{\text{default}}^*$.
- CSV schema remains structured with flattened parameter names:
  `game_id,is_default,<flattened_metaparams...>,objective_value`
  where `objective_value` is the posterior predictive mean or observed loss.

---

### 7. Validation Test Plan (`tests/test_mtbo.py`)
1. **Introspection & Broadcasting Test:** Verify that `NEEstimator.get_param_specs` returns valid shapes and that scalar search bounds correctly broadcast to per-player arrays.
2. **Fast Mock Active-Task Optimization Run:** Run a lightweight Swersky MTBO loop on 3 small quadratic games ($N=2, m=1$, $N_{\text{iter}}=30, K=10, n_{\text{init}}=3, n_{\text{bo_steps}}=3$) ensuring complete execution in $< 10$ seconds.
3. **Bound Containment:** Verify that all returned parameter dictionaries strictly lie within the bounds defined by `search_space`.
4. **CSV Export Fidelity:** Verify that the output CSV matches the returned in-memory dictionaries and columns exactly.

---

### 8. Relation to Benchmark Experiments
For specific notebook execution parameters, 100-game ensemble configuration, automated 5 representative games selection, and action-only visualization in `analysis.ipynb`, see [`specs/03_notebook_experiment.md`](file:///home/bruno/Documents/PayoffBasedNEComparison/specs/03_notebook_experiment.md).