# Specification 06: NESeeking Hyperparameter Dimension Reduction Study

## 1. Executive Summary & Problem Formulation
In high-dimensional hyperparameter optimization of continuous-time extremum seeking (`NESeeking`), treating all 8 parameters ($k_1, k_2, a_1, a_2, \omega_1, \omega_2, \omega_{h1}, \omega_{h2}$) as independent continuous variables in $[0.01, 100.0]^8$ induces severe failure modes:
1. **The Curse of Dimensionality:** An 8D space is vastly sparse. Quasi-random candidate pools of size 1,024 cannot effectively cover $[0, 1]^8$. Far from evaluated points, the Gaussian Process surrogate posterior reverts to its prior mean $\mu(x) \approx \bar{y} \gg y^*$, resulting in deeply negative standardized improvements $z = \frac{y^* - \mu(x)}{\sigma(x)} < -5$ and infinitesimal Expected Improvement ($\text{EI} < 10^{-12}$). This causes premature task pruning after only 1–2 BO steps.
2. **Computational Memory Explosion:** Exact Gaussian Processes (`MultiTaskGP`) scale as $\mathcal{O}(N^2)$ in memory and $\mathcal{O}(N^3)$ in time with $N = n_{\text{init}} \times M$. For $M = 100$ and $n_{\text{init}} = 256$, $N = 25,600$, requiring $> 30\text{ GB}$ of RAM during Cholesky factorization and autograd backpropagation, triggering Out-Of-Memory (OOM) crashes.
3. **Physical Redundancy:** Extremum seeking averaging theory (Frihauf et al. 2012, Krstic & Wang 2000) dictates that filter cutoffs $\omega_h$ only serve to wash out slow DC drift and do not affect averaged gradient dynamics within the passband $\omega_{\text{drift}} \ll \omega_h \ll \omega$. Furthermore, multi-agent dither frequencies must maintain frequency separation ($\omega_2 \neq \omega_1$) to prevent cross-demodulation interference.

This study systematically investigates and validates reducing the search space from **8D to 5D** using canonical extremum seeking rules:
$$\omega_2 = 1.35 \cdot \omega_1, \qquad \omega_{h1} = 0.20 \cdot \omega_1, \qquad \omega_{h2} = 0.20 \cdot \omega_2$$

---

## 2. Parameterization Spaces

### A. Unconstrained 8D Search Space ($\Theta_{\text{8D}}$)
* $k_1, k_2 \in [0.01, 100.0]$
* $a_1, a_2 \in [0.01, 100.0]$
* $\omega_1, \omega_2 \in [1.0, 50.0]$
* $\omega_{h1}, \omega_{h2} \in [0.1, 50.0]$
* Fixed: $\Delta t = 0.001$

### B. Coupled 5D Search Space ($\Theta_{\text{5D}}$)
* $k_1, k_2 \in [0.01, 100.0]$
* $a_1, a_2 \in [0.01, 100.0]$
* $\omega_1 \in [1.0, 50.0]$
* Coupled rules:
  $$\omega_2 = 1.35 \cdot \omega_1, \quad \omega_{h1} = 0.20 \cdot \omega_1, \quad \omega_{h2} = 0.20 \cdot \omega_2$$
* Fixed: $\Delta t = 0.001$

---

## 3. Study Experimental Design (`04_neseeking_dimension_reduction_study.ipynb`)

### Experiment 1: 5D vs. 8D MTBO Benchmark (5 Representative Games)
* Testbed: 5 representative games from `generate_quadratic_games(100)`:
  - Game 27: Extreme Ill-Conditioned
  - Game 30: Well-Behaved / Isotropic
  - Game 61: Narrow Action Bounds
  - Game 65: Highly Asymmetric Coupling
  - Game 67: Boundary-Adjacent Equilibrium
* Trajectory simulation: `num_steps = 5000` (5.0s, $\Delta t = 0.001$).
* MTBO setup: `n_init = 10`, `max_bo_steps = 25`, `eps_ei = 1e-4`, `patience = 6`, `tol_stagnation = 1e-4`.
* Metrics: Final regret loss ($\log_{10}$ Total Regret), wall-clock optimization runtime, evaluated BO steps before convergence, EI persistence.

### Experiment 2: One-At-A-Time (OAT) 1D Parameter Sensitivity Slices
* Using the winning optimal parameters as reference, individually sweep:
  1. $\omega_{h1} \in [0.01, 100.0]$ (50 log-spaced points) holding other 7 parameters fixed.
  2. $\omega_{h2} \in [0.01, 100.0]$ (50 log-spaced points) holding other 7 parameters fixed.
  3. $\omega_2 \in [0.01, 100.0]$ (50 log-spaced points) holding other 7 parameters fixed.
* Expected outcome: Wide flat invariant passband for $\omega_h \ll \omega_1$, sharp degradation cliff as $\omega_h \to \omega_1$, and resonant interference spikes when $\omega_2 \approx \omega_1$.

### Experiment 3: Global Sensitivity Analysis (Sobol Variance Decomposition)
* Computes First-Order ($S_i$) and Total-Order ($S_{T_i}$) Sobol Sensitivity Indices across all 8 parameters using Jansen (1999) / Saltelli (2010) estimators with scrambled Sobol sequence sampling.
* Mathematical validation: Proves that $\omega_{h1}, \omega_{h2}$ have near-zero total sensitivity ($S_T < 0.05$), confirming that fixing or coupling them does not sacrifice optimization performance.

### Experiment 4: Action Trajectory Convergence Overlays
* Simulates and plots action profiles $u_1(t), u_2(t)$ over 5.0s for all 5 representative games comparing 5D Optima vs. 8D Optima vs. True Nash Equilibrium.

### Experiment 5: Statistical Significance & Decision Framework
* Paired t-test and Wilcoxon signed-rank test comparing 5D vs. 8D regret losses across the representative games.
* Formal recommendation for full 100-game ensemble benchmarks.

---

## 4. Artifact Directory & Organization
All generated data, tables, and visualization plots are strictly co-located within:
`data/04_neseeking_reduction_study/`
* `mtbo_8d_results.csv`: MTBO results for 8D.
* `mtbo_5d_results.csv`: MTBO results for 5D.
* `mtbo_5d_vs_8d_comparison.csv`: Comparative performance metrics.
* `1d_sensitivity_curves.png`: 3-panel sensitivity slice curves.
* `sobol_sensitivity_indices.png`: Total-order Sobol index bar chart.
* `rep_game_{idx}_trajectory.png`: Action trajectory overlay plots.
* `reduction_study_summary.csv`: Comprehensive metrics table.
