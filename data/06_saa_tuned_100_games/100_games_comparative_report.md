# 100-Game Ensemble Benchmark: MTBO vs. Optuna Comparative Analysis Report

**Execution Date:** 2026-10-13 10:02:28

## 1. Executive Summary
This report benchmarks Multi-Task Bayesian Optimization (MTBO, Swersky et al. 2013) executed in a 4x25 batch architecture against dedicated single-game Optuna Tree-structured Parzen Estimator (TPE) studies across 100 strongly monotone quadratic games. The objective function is the Logarithmic Total Systemic Regret over 20,000 steps ($dt = 0.001$, $\varepsilon = 10^{-4}$ fixed).

## 2. Computational Timing Benchmark & Extrapolation

| Algorithm | MTBO Actual Time (100 Games) | Optuna Time (5 Games) | Optuna Extrapolated (100 Games) | MTBO Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **ZOGrandientPlay** | 24320.2 s (405.34 min) | 438.9 s (7.31 min) | 8777.2 s (2.44 h) | **0.36x** |
| **NESeeking5D** | 8197.2 s (136.62 min) | 401.2 s (6.69 min) | 8024.5 s (2.23 h) | **0.98x** |

### Evaluation Budgets & Early Stopping Summary:
- **ZOGrandientPlay MTBO (4x25):** Evaluated 4 batches with $n_{{\text{init}}}=64$ per batch (6400 initial observations, $K=5$ SAA) + 125 sequential BO steps across batches.
- **NESeeking5D MTBO (4x25):** Evaluated 4 batches with $n_{{\text{init}}}=64$ per batch (6400 initial observations) + 162 sequential BO steps across batches.

## 3. Representative Games Performance (Log10 Total Regret)

| Game | Category | True NE | ZO Optuna | ZO MTBO Tailored | ZO MTBO Consensus Default | NE Optuna | NE MTBO Tailored | NE MTBO Consensus Default |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game 11 | Extreme Ill-Conditioned (kappa=15.16) | [0.289, 0.074] | 1.7617 | 2.9226 | 2.8240 | 2.8884 | 2.3815 | 2.6927 |
| Game 70 | Well-Behaved / Isotropic (kappa=1.05) | [0.325, 0.416] | 2.3832 | 3.1819 | 2.6810 | 2.6884 | 2.7016 | 2.7711 |
| Game 97 | Narrow Action Bounds (vol=0.0818) | [-0.051, 0.048] | 1.1342 | 3.0779 | 2.9580 | 1.5869 | 1.8188 | 1.8435 |
| Game 40 | Highly Asymmetric Coupling (skew=2.781) | [0.165, 0.275] | 2.8067 | 2.9052 | 2.9795 | 3.0364 | 2.8896 | 2.9169 |
| Game 17 | Boundary-Adjacent Equilibrium (dist=0.006) | [0.191, 0.636] | 2.0186 | 2.1508 | 2.3872 | 2.0507 | 1.2919 | 2.3072 |

## 4. Surrogate Model Validation & Uncertainty Calibration

| Algorithm | LOO RMSE | Standardized MSE (SMSE) | Calibration Std | Anomalies (Default > Tailored) | Surrogate Refinements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ZOGrandientPlay** | 0.1161 | 0.0086 | 1.0926 | 1 | 100 |
| **NESeeking5D** | 0.1493 | 0.0148 | 0.9945 | 10 | 100 |

## 5. 100-Game Global Ensemble Statistics & Statistical Significance

| Metric | ZOGrandientPlay | NESeeking5D |
| :--- | :--- | :--- |
| **Tailored Optima (Mean +/- Std)** | 2.4532 +/- 0.3622 | 2.2061 +/- 0.3653 |
| **Tailored Optima (Median)** | 2.4501 | 2.2524 |
| **Consensus Default (Mean +/- Std)** | 2.5285 +/- 0.3370 | 2.4304 +/- 0.2525 |
| **Consensus Default (Median)** | 2.4698 | 2.4385 |
| **Mean Tunability Gain (Def - Tailored)** | 0.0753 | 0.2242 |
| **Tunability 95% Confidence Interval** | [-0.0038, 0.1544] | [0.1725, 0.2760] |
| **Max Tunability Gain** | 1.4950 | 1.5675 |
| **Paired t-test (p-value)** | 6.5173e-02 | 2.0596e-13 |
| **Wilcoxon Signed-Rank (p-value)** | 3.9168e-02 | 2.1427e-16 |

## 6. Strategic Recommendation
- **Recommendation: RECONSIDER OPTUNA.** Direct per-game Optuna tuning showed competitive or superior runtime and solution quality without surrogate modeling overhead.
