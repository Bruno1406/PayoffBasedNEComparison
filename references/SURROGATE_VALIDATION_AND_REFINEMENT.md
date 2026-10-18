# Surrogate Validation & Adaptive Refinement: An Intuitive Guide

This document explains the three core verification and refinement mechanisms used in our Multi-Task Bayesian Optimization (MTBO) pipeline:
1. **`loo_rmse`** (Leave-One-Out Root Mean Square Error)
2. **`uncertainty_calibration`** (Standardized Residual / $z$-Score Calibration)
3. **`adaptive_refinement`** (Post-Evaluation Basin Specialization for Anomalies)

It is written for engineers and researchers who want to understand *what these metrics mean, why they matter, and how to interpret their values*, without getting bogged down in unnecessary machine learning jargon.

---

## 1. Why Do We Need Validation in Bayesian Optimization?

In standard machine learning (like image classification or regression), we validate models by splitting data into a **training set** and a **test set**.

In Bayesian Optimization (BO), every data point comes from a **slow, expensive game simulation** (solving differential equations over thousands of time steps). We cannot afford to throw away 20% or 30% of our hard-earned simulations just to create an idle "test set".

This creates a critical question:
> *How can we verify that our Gaussian Process surrogate model is actually learning the true game dynamics, rather than hallucinating or overfitting, without running extra simulations?*

This is where **Leave-One-Out Cross-Validation** and **Uncertainty Calibration** come in.

---

## 2. Leave-One-Out RMSE (`loo_rmse`)

### The Intuition: "Hiding One Flashcard at a Time"
Imagine you are studying a deck of 50 flashcards. To test yourself honestly without buying a new deck:
1. You hide Card 1 behind your back.
2. Based only on the remaining 49 cards, you try to predict what is written on Card 1.
3. You check your error.
4. You put Card 1 back, hide Card 2, predict Card 2, and repeat this for all 50 cards.

Your average error across all 50 tests is your **Leave-One-Out (LOO) error**. Because the card being predicted was never seen during that specific prediction, it gives an unbiased measure of how well your model predicts **new, unseen data**.

```
[Evaluated Points: 1, 2, 3, ..., N]
 ├── Hide Point 1 ──> Train on [2, ..., N] ──> Predict Point 1 ──> Error e_1
 ├── Hide Point 2 ──> Train on [1, 3, ..., N] ──> Predict Point 2 ──> Error e_2
 └── ...
 └── Hide Point N ──> Train on [1, ..., N-1] ──> Predict Point N ──> Error e_N
```

### The Gaussian Process Shortcut: Free Computation
In most machine learning algorithms (like neural networks or gradient boosting), Leave-One-Out validation is too slow because you would have to re-train the model $N$ separate times.

In Gaussian Processes, however, there is a famous mathematical shortcut discovered in statistics: once the kernel covariance matrix $\mathbf{K}$ has been inverted during the initial model fit, **the Leave-One-Out prediction for every single point can be calculated in closed form with simple vector multiplication**:
$$\mu_{-i}(x_i) = y_i - \frac{[\mathbf{K}^{-1}\mathbf{y}]_i}{[\mathbf{K}^{-1}]_{ii}}$$

It requires **zero extra game simulations** and executes in less than **1 millisecond**!

### The Formula
From these individual prediction errors, the **LOO Root Mean Squared Error** is:
$$\text{RMSE}_{\text{LOO}} = \sqrt{\frac{1}{N_{\text{total}}} \sum_{i=1}^{N_{\text{total}}} \left( y_i - \mu_{-i}(x_i) \right)^2}$$

### How to Interpret the Value:
* **The Scale:** The RMSE is measured in the exact same units as our loss function (which in our project is $\log_{10}(\text{Regret})$).
* **$\text{RMSE}_{\text{LOO}} < 0.15$:** **Excellent fit.** The surrogate model predicts unvisited game performances with an error of less than $0.15$ on a logarithmic scale. The surrogate landscape is highly trustworthy.
* **$\text{RMSE}_{\text{LOO}} \in [0.15, 0.40]$:** **Acceptable fit.** The model captures the broad macro-trends across hyperparameters, though local fine-tuning may have mild inaccuracies.
* **$\text{RMSE}_{\text{LOO}} > 0.80$:** **Poor fit.** The model is struggling to capture the response surface, likely due to insufficient samples or noisy dynamics.

---

## 3. Uncertainty Calibration (`uncertainty_calibration`)

### The Intuition: "The Honest Weather Forecaster"
A standard optimization algorithm only predicts a single number (e.g., *"at $\theta = (1.5, 0.2)$, the expected regret is $-0.85$"*).

A Gaussian Process is special because it also predicts its own **uncertainty / confidence interval** (e.g., *$-0.85 \pm 0.10$*).

Bayesian Optimization relies completely on this uncertainty to make decisions:
* **Exploitation:** Sample where the predicted loss is low ($\mu$ is good).
* **Exploration:** Sample where the uncertainty is high ($\sigma$ is large) to discover unknown basins.

Now consider what happens if the model's uncertainty is broken:
* **Overconfident Model ($\sigma$ too small):** The model thinks it knows everything. It stops exploring and gets trapped in the first mediocre local minimum it encounters.
* **Underconfident Model ($\sigma$ too large):** The model thinks it knows nothing. It spends all its time exploring random unpromising corners of the space and refuses to converge.

**Calibration** checks whether the model's confidence estimates match real-world prediction errors.

### The Math: Standardized Residuals ($z$-Scores)
For every point $i$, we compare the actual prediction error to the model's reported standard deviation $\sigma_{-i}$:
$$z_i = \frac{y_i - \mu_{-i}(x_i)}{\sigma_{-i}(x_i)}$$

* $z_i = 0$ means the prediction was exact.
* $z_i = +1$ means the real value was 1 standard deviation higher than predicted.
* $z_i = -2$ means the real value was 2 standard deviations lower than predicted.

If the Gaussian Process is well-calibrated, the distribution of all $z$-scores should follow a **Standard Normal Distribution** $\mathcal{N}(0, 1)$, meaning:
$$\text{Mean}(z) \approx 0.0, \qquad \text{Std}(z) \approx 1.0$$

### How to Interpret `calibration_std`:
* **$\text{Std}(z) \approx 1.0$ (Ideal: $[0.8, 1.3]$):** **Perfect Calibration.** When the model says it is 95% confident, it is actually right 95% of the time. The exploration/exploitation balance of BO is mathematically sound.
* **$\text{Std}(z) > 1.5$ (Overconfident):** Real errors are larger than the model thought. The model's noise variance or lengthscales may be underestimated.
* **$\text{Std}(z) < 0.6$ (Underconfident):** Real errors are much smaller than the model thought. The model is over-cautious and may explore excessively.

---

## 4. Adaptive Refinement (`adaptive_refinement`)

### The Problem: What is a "Tunability Anomaly"?
In game theory and hyperparameter tuning:
* The **Universal Default Parameter ($\theta_{\text{def}}^*$)** is a single "one-size-fits-all" compromise parameter meant to work moderately well across all $M$ games.
* The **Tailored Parameter ($\theta_j^*$)** is supposed to be custom-tailored to maximize performance specifically on Game $j$.

By definition of optimization, **a custom solution should always beat (or at least match) a compromise solution**:
$$y_j(\theta_j^*) \le y_j(\theta_{\text{def}}^*)$$
The difference is the **Tunability Index** $d_j = y_j(\theta_{\text{def}}^*) - y_j^* \ge 0$, representing the performance gain achieved by tuning specifically for Game $j$.

An **Anomaly** occurs when:
$$y_j(\theta_{\text{def}}^*) < y_j(\theta_j^*) \implies d_j < 0$$
*The "compromise" parameter accidentally beat Game $j$'s "custom" parameter!*

### Why Does This Happen in Approach B?
In our high-efficiency Approach B ($M+1$ evaluations per step):
1. During the BO loop, candidate default parameters $\theta_{\text{avg}}^*$ are evaluated on **only 1 task per step** to save simulation time.
2. Game $j$ never evaluated candidate $\theta_{\text{def}}^*$ during its own 40 search steps. Game $j$'s acquisition function searched other regions of the 8-dimensional space and found its best point at, say, $-0.78$.
3. At the end of the run, we evaluate $\theta_{\text{def}}^*$ on all games to establish baseline default performance. That is when we discover: *Game $j$ actually scores $-0.80$ on the default parameter!*
4. **The revelation:** Game $j$'s individual search simply missed that basin during its budget!

### The Solution: Adaptive Refinement Burst
Instead of passively accepting an anomaly and artificially clamping $d_j = 0$ (a band-aid), **Adaptive Refinement uses the anomaly as an active optimization trigger**:

```
[BO Run Ends] ──> Evaluate Default on all M Games
                     │
                     ├── Does y_j(theta_def) >= y_j* for all games? 
                     │   └── YES: Perfect! All tunabilities d_j >= 0. Finished.
                     │
                     └── NO: Anomaly detected on Game j!
                         │
                         ├── 1. Append (theta_def, y_j(theta_def)) to Game j's data
                         ├── 2. Launch 4-step local BO refinement burst on Game j
                         │      (centered around theta_def)
                         └── 3. Game j discovers specialized point theta_j* <= theta_def
                                └── Tunability d_j = y_def - y_j* > 0 (Genuine Gain!)
```

### Why Adaptive Refinement is So Efficient:
* In our 20-game benchmark, **only 1 out of 36 games (2.7%)** ever had an anomaly.
* Rather than running hundreds of extra simulations across all games during the loop, we only spend **4 extra simulation steps on the single game that needs it**.
* Game $j$ uses the default parameter as a springboard, refines the parameters specifically for its own payoff matrix, and discovers its true tailored optimum.

---

## 5. Summary & Quick Reference Cheat Sheet

| Metric / Feature | What It Measures | Ideal Target | What It Means If Bad |
| :--- | :--- | :--- | :--- |
| **`loo_rmse`** | Surrogate prediction error on unvisited parameter combinations. | $< 0.15$ (Log-regret scale) | $> 0.50$: Surrogate landscape is noisy or under-sampled. |
| **`loo_smse`** | Relative variance explained by the surrogate model. | $\ll 1.0$ (e.g. $0.05\text{--}0.20$) | $\ge 1.0$: Surrogate is no better than predicting the mean. |
| **`calibration_std`** | Reliability of the GP's reported uncertainty estimates. | $\approx 1.0$ ($[0.8, 1.3]$) | $> 1.5$: Model is overconfident (risks getting trapped). <br> $< 0.6$: Model is underconfident (explores too much). |
| **`adaptive_refinement`** | Automatic local search burst triggered if default beats tailored. | $0$ anomalies remaining after refinement. | Eliminates negative tunability ($d_j \ge 0$) with genuine local specialization. |
