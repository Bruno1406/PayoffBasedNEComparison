# 100-Game Ensemble Benchmark: MTBO vs. Optuna Comparative Analysis Report

**Execution Date:** 2026-10-09 23:50:23

## 1. Executive Summary
This report benchmarks Multi-Task Bayesian Optimization (MTBO, Swersky et al. 2013) against dedicated Optuna Tree-structured Parzen Estimator (TPE) studies across 100 strongly monotone quadratic games. The objective function is the Logarithmic Total Systemic Regret over 10,000 steps ($dt = 0.001$, $\varepsilon = 10^{-4}$ fixed).

## 2. Computational Timing Benchmark & Extrapolation

| Algorithm | MTBO Actual Time (100 Games) | Optuna Time (5 Games) | Optuna Extrapolated (100 Games) | MTBO Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **ZOGrandientPlay** | 1529.2 s (25.49 min) | 100.3 s (1.67 min) | 2006.3 s (0.56 h) | **1.31x** |
| **NESeeking** | 5279.3 s (87.99 min) | 82.6 s (1.38 min) | 1652.0 s (0.46 h) | **0.31x** |

### Evaluation Budgets & Early Stopping Summary:
- **ZOGrandientPlay MTBO:** Evaluated 100 initial samples + 500 sequential BO steps. Stop reason: `max_bo_steps_reached` (Early stopped: `False`).
- **NESeeking MTBO:** Evaluated 100 initial samples + 684 sequential BO steps. Stop reason: `ei_convergence` (Early stopped: `True`).

## 3. Representative Games Performance (Log10 Total Regret)

| Game | Category | True NE | ZO Optuna | ZO MTBO Tailored | ZO MTBO Default | NE Optuna | NE MTBO Tailored | NE MTBO Default |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game 67 | Extreme Ill-Conditioned (kappa=20.02) | [0.969, -0.089] | 2.9808 | 2.9757 | 3.2929 | 3.2056 | 1.7359 | 3.3885 |
| Game 27 | Well-Behaved / Isotropic (kappa=1.06) | [-0.563, -0.386] | 2.2158 | 2.5157 | 2.8982 | 2.6798 | 2.3271 | 2.4854 |
| Game 65 | Narrow Action Bounds (vol=0.1897) | [-0.015, 0.312] | 2.7579 | 2.7695 | 2.3350 | 1.5382 | 1.6896 | 1.6971 |
| Game 61 | Highly Asymmetric Coupling (skew=2.715) | [0.566, -0.171] | 3.4025 | 3.6151 | 3.4617 | 2.1047 | 2.1270 | 2.1491 |
| Game 30 | Boundary-Adjacent Equilibrium (dist=0.001) | [0.468, -0.617] | 3.2173 | 3.4727 | 3.1461 | 2.3747 | 2.1233 | 2.1209 |

## 4. 100-Game Global Ensemble Statistics

| Metric | ZOGrandientPlay | NESeeking |
| :--- | :--- | :--- |
| **Tailored Optima (Mean +/- Std)** | 2.9791 +/- 0.4389 | 2.6244 +/- 0.3901 |
| **Tailored Optima (Median)** | 3.1039 | 2.7027 |
| **Universal Default (Mean +/- Std)** | 2.8252 +/- 0.5778 | 2.6996 +/- 0.4306 |
| **Universal Default (Median)** | 2.8872 | 2.8035 |
| **Mean Tunability Gain (Def - Tailored)** | -0.1539 | 0.0752 |
| **Max Tunability Gain** | 1.2217 | 1.6526 |

## 5. Strategic Recommendation
- **Recommendation: RECONSIDER OPTUNA.** Direct per-game Optuna tuning showed competitive or superior runtime and solution quality without surrogate modeling overhead.
