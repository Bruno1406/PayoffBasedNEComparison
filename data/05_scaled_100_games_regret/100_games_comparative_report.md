# 100-Game Ensemble Benchmark: MTBO vs. Optuna Comparative Analysis Report

**Execution Date:** 2026-10-12 20:14:33

## 1. Executive Summary
This report benchmarks Multi-Task Bayesian Optimization (MTBO, Swersky et al. 2013) executed in a 4x25 batch architecture against dedicated single-game Optuna Tree-structured Parzen Estimator (TPE) studies across 100 strongly monotone quadratic games. The objective function is the Logarithmic Total Systemic Regret over 20,000 steps ($dt = 0.001$, $\varepsilon = 10^{-4}$ fixed).

## 2. Computational Timing Benchmark & Extrapolation

| Algorithm | MTBO Actual Time (100 Games) | Optuna Time (5 Games) | Optuna Extrapolated (100 Games) | MTBO Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **ZOGrandientPlay** | 2488.5 s (41.47 min) | 474.5 s (7.91 min) | 9490.6 s (2.64 h) | **3.81x** |
| **NESeeking5D** | 3218.0 s (53.63 min) | 480.5 s (8.01 min) | 9610.5 s (2.67 h) | **2.99x** |

### Evaluation Budgets & Early Stopping Summary:
- **ZOGrandientPlay MTBO (4x25):** Evaluated 4 batches with $n_{{\text{init}}}=16$ per batch (1600 initial observations) + 118 sequential BO steps across batches.
- **NESeeking5D MTBO (4x25):** Evaluated 4 batches with $n_{{\text{init}}}=32$ per batch (3200 initial observations) + 117 sequential BO steps across batches.

## 3. Representative Games Performance (Log10 Total Regret)

| Game | Category | True NE | ZO Optuna | ZO MTBO Tailored | ZO MTBO Consensus Default | NE Optuna | NE MTBO Tailored | NE MTBO Consensus Default |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game 11 | Extreme Ill-Conditioned (kappa=15.16) | [0.289, 0.074] | 1.7458 | 2.6101 | 3.3292 | 2.8884 | 2.3585 | 2.7387 |
| Game 70 | Well-Behaved / Isotropic (kappa=1.05) | [0.325, 0.416] | 2.3560 | 2.4773 | 2.6893 | 2.6884 | 2.7156 | 2.7952 |
| Game 97 | Narrow Action Bounds (vol=0.0818) | [-0.051, 0.048] | 1.1990 | 1.3402 | 3.0071 | 1.5869 | 1.8103 | 1.8678 |
| Game 40 | Highly Asymmetric Coupling (skew=2.781) | [0.165, 0.275] | 2.2636 | 3.3116 | 2.9346 | 3.0364 | 2.8589 | 2.9527 |
| Game 17 | Boundary-Adjacent Equilibrium (dist=0.006) | [0.191, 0.636] | 2.0858 | 2.0792 | 2.3046 | 2.0507 | 1.4298 | 2.2334 |

## 4. Surrogate Model Validation & Uncertainty Calibration

| Algorithm | LOO RMSE | Standardized MSE (SMSE) | Calibration Std | Anomalies (Default > Tailored) | Surrogate Refinements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ZOGrandientPlay** | 0.5724 | 0.2492 | 1.5269 | 6 | 100 |
| **NESeeking5D** | 0.1995 | 0.0305 | 1.0025 | 18 | 100 |

## 5. 100-Game Global Ensemble Statistics & Statistical Significance

| Metric | ZOGrandientPlay | NESeeking5D |
| :--- | :--- | :--- |
| **Tailored Optima (Mean +/- Std)** | 2.5288 +/- 0.4022 | 2.2526 +/- 0.3557 |
| **Tailored Optima (Median)** | 2.5385 | 2.3135 |
| **Consensus Default (Mean +/- Std)** | 2.5982 +/- 0.3947 | 2.4333 +/- 0.2501 |
| **Consensus Default (Median)** | 2.5552 | 2.4395 |
| **Mean Tunability Gain (Def - Tailored)** | 0.0694 | 0.1808 |
| **Tunability 95% Confidence Interval** | [-0.0257, 0.1644] | [0.1335, 0.2281] |
| **Max Tunability Gain** | 1.6670 | 1.4522 |
| **Paired t-test (p-value)** | 1.5573e-01 | 2.8174e-11 |
| **Wilcoxon Signed-Rank (p-value)** | 5.1575e-02 | 5.0345e-13 |

## 6. Strategic Recommendation
- **Recommendation: RETAIN MTBO.** MTBO demonstrates superior computational efficiency through cross-task transfer learning, finding both game-specific optima and universal consensus defaults simultaneously while saving substantial compute compared to 100 independent Optuna runs.
