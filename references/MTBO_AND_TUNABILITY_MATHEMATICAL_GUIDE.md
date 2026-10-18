# Technical and Mathematical Guide: Multi-Task Bayesian Optimization and the Tunability Index in Payoff-Based Game Learning

---

## 1. Introduction to Gaussian Processes, Single-Task Bayesian Optimization, Multi-Dimensional GPs, and MTBO

### 1.1 Gaussian Processes as Non-Parametric Function Priors
In black-box optimization, the objective function $f: \mathcal{X} \to \mathbb{R}$ lacks an analytical closed-form representation and gradient evaluations $\nabla f(x)$ are unavailable. A **Gaussian Process (GP)** provides a non-parametric Bayesian framework that places a prior probability distribution directly over the space of functions:

$$f(x) \sim \mathcal{GP}\left(m(x), \; k(x, x')\right)$$

where:
- $m(x) = \mathbb{E}[f(x)]$ is the **prior mean function**, conventionally set to $m(x) = 0$ or the constant empirical mean of initial observations.
- $k(x, x') = \text{Cov}(f(x), f(x'))$ is the **covariance kernel**, encoding structural properties such as smoothness, stationarity, characteristic lengthscales, and amplitude.

A prominent choice for continuous physical systems is the **Matérn 5/2 kernel**, which models functions that are twice mean-square differentiable:

$$k_{\text{Matérn 5/2}}(x, x') = \sigma_f^2 \left( 1 + \sqrt{5} r + \frac{5}{3} r^2 \right) \exp\left(-\sqrt{5} r\right), \quad r = \sqrt{\sum_{i=1}^d \frac{(x_i - x'_i)^2}{\ell_i^2}}$$

where $\sigma_f^2$ is the output signal variance and $\ell_i > 0$ denotes the Automatic Relevance Determination (ARD) lengthscale along dimension $i$.

#### Derivation of the Posterior Distribution via Gaussian Conditioning
Suppose we collect a set of $N$ noisy observations:
$$\mathcal{D}_N = \left\{ (x_n, y_n) \right\}_{n=1}^N, \quad y_n = f(x_n) + \epsilon_n, \quad \epsilon_n \sim \mathcal{N}(0, \sigma_n^2)$$
Let $\mathbf{X} = [x_1, \dots, x_N]^T \in \mathbb{R}^{N \times d}$ and $\mathbf{y} = [y_1, \dots, y_N]^T \in \mathbb{R}^N$. By the definition of a Gaussian Process, any finite collection of function evaluations is jointly Gaussian. For an unobserved candidate point $x_* \in \mathcal{X}$, the joint prior distribution between observed targets $\mathbf{y}$ and latent target $f_* = f(x_*)$ is:

$$\begin{bmatrix} \mathbf{y} \\ f_* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{m}(\mathbf{X}) \\ m(x_*) \end{bmatrix}, \; \begin{bmatrix} \mathbf{K}(\mathbf{X}, \mathbf{X}) + \sigma_n^2 \mathbf{I}_N & \mathbf{k}(x_*, \mathbf{X})^T \\ \mathbf{k}(x_*, \mathbf{X}) & k(x_*, x_*) \end{bmatrix} \right)$$

where:
- $\mathbf{K} \in \mathbb{R}^{N \times N}$ has entries $\mathbf{K}_{ij} = k(x_i, x_j)$.
- $\mathbf{k}_* = [k(x_*, x_1), \dots, k(x_*, x_N)]^T \in \mathbb{R}^N$.
- $\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I}_N$ is the regularized Gram matrix.

Applying the **conditional probability rule for partitioned multivariate normal distributions** (the Schur complement formulation), the conditional distribution $p(f_* \mid \mathbf{X}, \mathbf{y}, x_*)$ remains strictly Gaussian:

$$f_* \mid \mathcal{D}_N, x_* \sim \mathcal{N}\left(\mu(x_*), \; \sigma^2(x_*)\right)$$

where the posterior predictive mean $\mu(x_*)$ and posterior predictive variance $\sigma^2(x_*)$ are given in closed form:

$$\mu(x_*) = m(x_*) + \mathbf{k}_*^T \mathbf{K}_y^{-1} \left( \mathbf{y} - \mathbf{m}(\mathbf{X}) \right)$$

$$\sigma^2(x_*) = k(x_*, x_*) - \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{k}_*$$

The posterior mean $\mu(x_*)$ acts as an interpolating predictor (a weighted linear combination of kernel evaluations), while the posterior variance $\sigma^2(x_*)$ quantifies the **epistemic uncertainty** stemming from lack of observations near $x_*$.

---

### 1.2 Single-Task Bayesian Optimization (STBO)
When optimizing an expensive black-box objective $f(x)$ over a bounded continuous domain $\mathcal{X}$, STBO constructs a sequential feedback loop:
1. Initialize with a design of experiments (e.g., Sobol quasi-random sampling).
2. Fit a Gaussian Process surrogate to all evaluated points $\mathcal{D}_N$.
3. Maximize an analytical **acquisition function** $\alpha(x; \mathcal{D}_N)$ that balances exploitation ($\mu(x)$ is favorable) and exploration ($\sigma(x)$ is high):
   $$x_{N+1} = \arg\max_{x \in \mathcal{X}} \alpha(x; \mathcal{D}_N)$$
4. Evaluate the expensive black-box simulator at $x_{N+1}$, augment $\mathcal{D}_{N+1} = \mathcal{D}_N \cup \{(x_{N+1}, y_{N+1})\}$, and repeat until the evaluation budget is exhausted.

---

### 1.3 Multi-Dimensional Gaussian Processes & Multi-Task Bayesian Optimization (MTBO)
In game-theoretic parameter tuning, an algorithm must perform across an ensemble of $M$ distinct games $\{G_1, G_2, \dots, G_M\}$, each defining a distinct objective function:
$$f_m(x) \triangleq L(x; G_m), \quad m \in \{1, \dots, M\}$$

Running $M$ completely independent Single-Task Bayesian Optimization studies discards the shared mathematical structure across the games. Conversely, averaging the games into a single scalar objective before surrogate modeling hides game-specific landscapes and prevents the discovery of tailored parameters.

**Multi-Task Bayesian Optimization (MTBO, Swersky et al. 2013)** models all $M$ response surfaces simultaneously via a vector-valued Gaussian Process:
$$\mathbf{f}(x) = \begin{bmatrix} f_1(x) \\ \vdots \\ f_M(x) \end{bmatrix} \sim \mathcal{GP}\left(\mathbf{m}(x), \; \mathbf{K}_{\text{joint}}(x, x')\right)$$

This allows cross-task transfer: an evaluation on Game $j$ updates the posterior beliefs across all other games correlated with Game $j$.

---

## 2. Mathematical Formulations: Multi-Task Modeling, Acquisition, and Tunability

### 2.1 The Latent Variable Model and the Intrinsic Coregionalization Model (ICM)
To construct a valid, positive-definite multi-task covariance kernel, Swersky et al. (2013) adopt the **Intrinsic Coregionalization Model (ICM)** based on linear latent variables.

#### The Latent Space Formulation
Assume each task function $f_m(x)$ is a linear combination of $Q \le M$ independent, unobserved scalar latent Gaussian Processes $u_q(x)$:
$$f_m(x) = \sum_{q=1}^Q w_{mq} u_q(x), \quad u_q(x) \sim \mathcal{GP}\left(0, \; k_X(x, x')\right), \quad \text{Cov}(u_q(x), u_{q'}(x')) = \delta_{qq'} k_X(x, x')$$

where $\mathbf{W} \in \mathbb{R}^{M \times Q}$ is a weight matrix whose entry $w_{mq}$ reflects the loading of task $m$ onto latent process $q$.

#### The Separable Covariance Tensor
The cross-covariance between task $m$ evaluated at input $x$ and task $m'$ evaluated at input $x'$ decomposes into the product of a **task covariance** and an **input space covariance**:

$$\text{Cov}\left(f_m(x), f_{m'}(x')\right) = \mathbb{E}\left[ \left(\sum_{q=1}^Q w_{mq} u_q(x)\right) \left(\sum_{q'=1}^Q w_{m'q'} u_{q'}(x')\right) \right] = \left(\sum_{q=1}^Q w_{mq} w_{m'q}\right) k_X(x, x')$$

In matrix notation, defining the positive semi-definite task covariance matrix:
$$\mathbf{K}_T = \mathbf{W}\mathbf{W}^T + \text{diag}(\boldsymbol{\kappa}) \in \mathbb{R}^{M \times M}$$
where $\boldsymbol{\kappa} \in \mathbb{R}_{\ge 0}^M$ captures task-specific independent noise/uniqueness, the complete multi-task covariance structure over a joint dataset takes the form of a **Kronecker product**:

$$\mathbf{K}_{\text{joint}} = \mathbf{K}_T \otimes \mathbf{K}_X$$

where:
- $\mathbf{K}_T \in \mathbb{R}^{M \times M}$ specifies inter-task correlations.
- $\mathbf{K}_X \in \mathbb{R}^{N \times N}$ specifies spatial correlations in the hyperparameter search space.

#### Learning the Inter-Task Correlation Structure
The entries of $\mathbf{W}$, lengthscales $\ell$, and observation noise variances are jointly estimated from data via **Type-II Maximum Likelihood Estimation (MLE)** by maximizing the Marginal Log-Likelihood (MLL):

$$\log p(\mathbf{y} \mid \mathbf{X}, \Theta_{\text{GP}}) = -\frac{1}{2} \mathbf{y}^T \mathbf{K}_{\text{joint}}^{-1} \mathbf{y} - \frac{1}{2} \log |\mathbf{K}_{\text{joint}}| - \frac{N M}{2} \log(2\pi)$$

Once $\mathbf{K}_T$ is learned, the pairwise correlation coefficient between Game $i$ and Game $j$ is quantified by:
$$\rho_{ij} = \frac{[\mathbf{K}_T]_{ij}}{\sqrt{[\mathbf{K}_T]_{ii} [\mathbf{K}_T]_{jj}}} \in [-1, 1]$$
- If $\rho_{ij} \to +1$, games share identical landscape topography; knowledge transfers fully.
- If $\rho_{ij} \to 0$, games are statistically orthogonal; no cross-task borrowing occurs.
- If $\rho_{ij} < 0$, parameters beneficial to Game $i$ degrade Game $j$.

---

### 2.2 The Expected Improvement (EI) Acquisition Function

To optimize an objective function $f(x)$ for minimization (e.g., minimizing cumulative systemic regret), we define the improvement over the current best-observed incumbent value $y_{\text{best}} = \min_{n=1,\dots,N} y_n$:

$$I(x) = \max\left(0, \; y_{\text{best}} - f(x)\right)$$

Because $f(x)$ is unknown, $I(x)$ is a random variable. The **Expected Improvement (EI)** acquisition function computes the expectation under the Gaussian Process posterior predictive distribution $f(x) \sim \mathcal{N}\left(\mu(x), \sigma^2(x)\right)$:

$$\text{EI}(x) \triangleq \mathbb{E}\left[ I(x) \mid \mathcal{D}_N \right] = \int_{-\infty}^{\infty} \max\left(0, \; y_{\text{best}} - f\right) \frac{1}{\sqrt{2\pi}\sigma(x)} \exp\left(-\frac{(f - \mu(x))^2}{2\sigma^2(x)}\right) df$$

#### Analytical Derivation
Split the integral at the boundary $f = y_{\text{best}}$, noting that the integrand vanishes for $f > y_{\text{best}}$:

$$\text{EI}(x) = \int_{-\infty}^{y_{\text{best}}} (y_{\text{best}} - f) \frac{1}{\sqrt{2\pi}\sigma(x)} \exp\left(-\frac{(f - \mu(x))^2}{2\sigma^2(x)}\right) df$$

Substitute the standardized change of variable $u = \frac{f - \mu(x)}{\sigma(x)}$, where $df = \sigma(x) du$. As $f \to -\infty$, $u \to -\infty$. At the upper limit $f = y_{\text{best}}$, the upper bound is the **standardized improvement**:

$$z \triangleq \frac{y_{\text{best}} - \mu(x)}{\sigma(x)}$$

The term $(y_{\text{best}} - f)$ becomes:
$$y_{\text{best}} - (\mu(x) + \sigma(x)u) = (y_{\text{best}} - \mu(x)) - \sigma(x)u = \sigma(x)(z - u)$$

Substituting these into the integral yields:

$$\text{EI}(x) = \int_{-\infty}^z \sigma(x)(z - u) \phi(u) \, du = \sigma(x) z \int_{-\infty}^z \phi(u) \, du - \sigma(x) \int_{-\infty}^z u \phi(u) \, du$$

where $\phi(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}$ is the standard normal probability density function (PDF).
By standard calculus:
1. $\int_{-\infty}^z \phi(u) \, du = \Phi(z)$, the standard normal cumulative distribution function (CDF).
2. Because $\frac{d}{du}[-\phi(u)] = u \phi(u)$, the second integral evaluates to:
   $$\int_{-\infty}^z u \phi(u) \, du = \left[ -\phi(u) \right]_{-\infty}^z = -\phi(z)$$

Combining these yields the famous closed-form formula:

$$\text{EI}(x) = \sigma(x) z \Phi(z) - \sigma(x) (-\phi(z)) = (y_{\text{best}} - \mu(x))\Phi(z) + \sigma(x)\phi(z)$$

$$\boxed{\text{EI}(x) = (y_{\text{best}} - \mu(x)) \Phi\left(\frac{y_{\text{best}} - \mu(x)}{\sigma(x)}\right) + \sigma(x) \phi\left(\frac{y_{\text{best}} - \mu(x)}{\sigma(x)}\right)}$$

#### Mathematical Intuition: The Exploration-Exploitation Duality
The resulting formula decomposes into two distinct mathematical terms:
1. **The Exploitation Term: $(y_{\text{best}} - \mu(x)) \Phi(z)$**
   Dominates when the predicted mean $\mu(x)$ is significantly lower than $y_{\text{best}}$ ($z > 0$). It pulls candidate queries toward regions where the surrogate expects low regret with high probability $\Phi(z)$.
2. **The Exploration Term: $\sigma(x) \phi(z)$**
   Dominates when epistemic uncertainty $\sigma(x)$ is large. The normal density $\phi(z)$ is non-negative and symmetric around $z = 0$, pulling candidate queries toward uncharted regions of the parameter space where lack of data leaves open the possibility of a superior local basin.

#### Multi-Task Extension
In our pipeline, the acquisition engine evaluates $M+1$ distinct Expected Improvement objectives simultaneously at each step:
- **$M$ Task-Specific Objectives:** $\text{EI}_j(x)$ using $y_{\text{best}, j}$ and posterior marginals $\mu_j(x), \sigma_j(x)$ to specialize parameters for individual games.
- **One Ensemble-Average Objective:** $\text{EI}_{\text{avg}}(x)$ over the surrogate ensemble mean $\bar{\mu}(x) = \frac{1}{M}\sum_{j=1}^M \mu_j(x)$ to guide the search toward the universal consensus default.

---

### 2.3 The Tunability Index Framework (Probst, Boulesteix, & Bischl 2019)

A central objective of this research is answering: *Does a payoff-based learning algorithm require game-specific parameter tuning, or does a single robust universal default suffice?*

Following the statistical definitions of Probst et al. (2019):
1. **The Universal Consensus Default ($\theta_{\text{default}}^*$):**
   The single fixed parameter configuration that minimizes average regret across the entire domain distribution $\mathcal{D}$:
   $$\theta_{\text{default}}^* \triangleq \arg\min_{\theta \in \Theta} \mathbb{E}_{G \sim \mathcal{D}} [L(\theta; G)] \approx \arg\min_{\theta \in \Theta} \frac{1}{M} \sum_{j=1}^M L(\theta; G_j)$$
2. **The Tailored Optimum ($\theta_j^*$):**
   The game-specific parameter vector optimizing performance on Game $j$ in isolation:
   $$\theta_j^* \triangleq \arg\min_{\theta \in \Theta} L(\theta; G_j)$$
3. **The Game-Specific Tunability Index ($\Delta_j$):**
   The performance gap on Game $j$ between the universal default and the customized optimum:
   $$\Delta_j \triangleq L(\theta_{\text{default}}^*; G_j) - L(\theta_j^*; G_j)$$
   By definition of optimality, $\Delta_j \ge 0$ in the absence of optimization error.
4. **The Overall Algorithmic Tunability ($\tau$):**
   The population expectation of the tunability index:
   $$\tau \triangleq \mathbb{E}_{G \sim \mathcal{D}} [\Delta(G)] \approx \frac{1}{M} \sum_{j=1}^M \Delta_j$$

#### Statistical Hypothesis Testing
To test whether hyperparameter tuning provides a statistically meaningful advantage over the consensus default, we formulate:
$$H_0: \tau = 0 \quad \text{vs.} \quad H_1: \tau > 0$$
- If $H_0$ cannot be rejected ($p \ge 0.05$), the algorithm is **untunable**: a practitioner should deploy the consensus default without wasting compute on custom tuning.
- If $H_0$ is strongly rejected ($p \ll 0.01$), the algorithm is **highly tunable**: performance depends critically on customizing parameters to individual game topologies.

---

## 3. Dimensionality Limits of MTBO and the NESeeking Dimension Reduction Study

### 3.1 The Curse of Dimensionality in Multi-Task Bayesian Optimization
While Gaussian Processes are powerful in low dimensions ($d \le 4$), scaling MTBO to higher dimensions ($d \ge 6$) encounters severe mathematical barriers:

#### 1. Geometric Sparsity and Distance Expansion
In a $d$-dimensional hypercube $[0, 1]^d$, the volume of a local neighborhood of radius $\epsilon$ shrinks exponentially: $\text{Vol} \propto \epsilon^d$. Conversely, to maintain a constant sampling density, the required number of evaluations scales as $\mathcal{O}(2^d)$.

#### 2. Mean Reversion and Acquisition Collapse
Because GP kernels rely on Euclidean distance $r^2 = \sum_{i=1}^d \frac{(x_i - x'_i)^2}{\ell_i^2}$, any unobserved candidate in a sparse high-dimensional space lies far from existing data points ($r \gg \ell$).
In regions far from observations:
$$\lim_{r \to \infty} \mathbf{k}_* \to 0 \implies \mu(x) \to m_0, \quad \sigma^2(x) \to k(x, x) = \sigma_f^2$$
If the optimization has already discovered a good incumbent ($y_{\text{best}} \ll m_0$), the standardized improvement becomes deeply negative:
$$z = \frac{y_{\text{best}} - m_0}{\sigma_f} \ll -3$$
In this regime, both the normal CDF $\Phi(z)$ and PDF $\phi(z)$ evaluate to near-zero ($< 10^{-6}$), causing:
$$\text{EI}(x) \longrightarrow 0 \quad \forall x \in \text{unexplored regions}$$
This causes **premature acquisition collapse**: the algorithm cannot find any candidate with $\text{EI} > \varepsilon_{\text{EI}}$, falsely concluding that the space has converged and terminating after only 1–2 BO steps.

#### 3. Cubic Matrix Scaling ($\mathcal{O}(N^3)$)
Exact Multi-Task GPs require solving linear systems with the joint covariance matrix $\mathbf{K}_{\text{joint}} \in \mathbb{R}^{NM \times NM}$. Evaluating $M = 100$ games with $N = 256$ evaluations yields a matrix of dimension $25,600 \times 25,600$. Computing its Cholesky decomposition requires:
$$\frac{1}{3} (NM)^3 \approx \frac{1}{3} (25,600)^3 \approx 5.6 \times 10^{12} \text{ FLOPs}$$
consuming over $30\text{ GB}$ of RAM and triggering operating system out-of-memory terminations.

---

### 3.2 The NESeeking Dimension Reduction Study: 8D to 5D

Continuous-time extremum seeking (`NESeeking`) implements nonlinear perturb-and-demodulate dynamics for each player $i$:
$$\dot{x}_i(t) = k_i (J_i(u(t)) - \eta_i(t)) M_i(t)$$
$$\dot{\eta}_i(t) = \omega_{hi} (J_i(u(t)) - \eta_i(t))$$
$$u_i(t) = x_i(t) + a_i \sin(\omega_i t), \quad M_i(t) = \frac{2}{a_i} \sin(\omega_i t)$$

In a 2-player game, an unconstrained parameterization yields **8 independent continuous parameters**:
$$\Theta_{\text{8D}} = \left\{ k_1, k_2, a_1, a_2, \omega_1, \omega_2, \omega_{h1}, \omega_{h2} \right\} \in [0.01, 100.0]^8$$

#### Control-Theoretic Rationale for Parameter Coupling (Averaging Theory)
According to the averaging analysis of extremum seeking systems (Frihauf, Krstic, & Basar 2012; Krstic & Wang 2000):
1. **Timescale Separation:** The washout filter state $\eta_i(t)$ acts as a high-pass filter that removes slow DC trend offsets from the payoff $J_i(u(t))$ without introducing phase lag at the perturbation frequency $\omega_i$. Its cutoff frequency $\omega_{hi}$ must simply lie in the intermediate passband:
   $$\omega_{\text{drift}} \ll \omega_{hi} \ll \omega_i$$
   As long as $\omega_{hi}$ is proportionally locked below $\omega_i$, its exact value does not alter the averaged gradient dynamics.
2. **Frequency Orthogonality:** To prevent cross-player demodulation interference (inter-modulatory limit cycles), player perturbation frequencies must be distinct: $\omega_1 \neq \omega_2$.

This motivated coupling the 8 parameters down to a **5-dimensional continuous search space ($\Theta_{\text{5D}}$)** via canonical geometric ratios:
$$\boxed{\omega_2 = 1.35 \cdot \omega_1, \qquad \omega_{h1} = 0.20 \cdot \omega_1, \qquad \omega_{h2} = 0.20 \cdot \omega_2}$$
$$\Theta_{\text{5D}} = \left\{ k_1, k_2, a_1, a_2, \omega_1 \right\}$$

#### Empirical Validation & Findings of the Dimension Study
To formally justify this reduction, a dedicated study was executed across 5 representative game geometries:

```text
================================================================================
                    8D vs. 5D DIMENSION REDUCTION BENCHMARK
================================================================================
Metric                           8D (Unconstrained)    5D (Coupled)    Impact
Mean Regret Loss (log10)              1.9302             1.7887       -0.1415 (5D wins)
Linear Regret Equivalent (10^Loss)    85.15              61.47        27.8% reduction
Cross-Game Standard Deviation        ±0.4631            ±0.2785       40.0% lower variance
Active BO Steps Evaluated               12                 12         Identical depth
================================================================================
```

The study established three decisive conclusions:
1. **Superior Performance:** Rather than suffering an expressive penalty, 5D achieved a **$27.8\%$ lower linear regret** than 8D.
2. **Elimination of Destructive Parameter Regimes:** In 8D, MTBO frequently sampled uncoordinated parameters where $\omega_1 \approx \omega_2$ or $\omega_h \ge \omega$, destabilizing the seeking loop. In 5D, timescale separation and frequency orthogonality are guaranteed by construction.
3. **Global Sensitivity Decomposition (Sobol Analysis):** Saltelli/Jansen variance decomposition proved that the total-order sensitivity indices for filter cutoffs $\omega_{h1}, \omega_{h2}$ were negligible ($S_T < 0.03$), confirming that fixing their ratios sacrifices zero optimization capacity.

---

## 4. Acquisition Optimization: Quasi-Random Sobol Sampling vs. Gradient Ascent

Once the Gaussian Process surrogate is fitted, finding the next evaluation point requires solving:
$$x_{N+1} = \arg\max_{x \in \mathcal{X}} \text{EI}(x)$$

Because $\text{EI}(x)$ is a mathematical construct evaluated entirely on the surrogate model without running physical game simulations, optimizing it should be fast and reliable. Two primary paradigms exist: **continuous gradient ascent** and **quasi-random pool sampling**.

### 4.1 Why Continuous Gradient Ascent Struggles on the EI Surface
Gradient ascent (e.g., L-BFGS-B initialized from random points) uses analytical gradients $\nabla_x \text{EI}(x)$. However, on complex multi-task landscapes, it faces fundamental limitations:
1. **Vanishing Gradients in Low-Uncertainty Basins:**
   $$\nabla_x \text{EI}(x) = -\nabla \mu(x) \Phi(z) + \nabla \sigma(x) \phi(z)$$
   In vast sub-optimal regions where $z = \frac{y_{\text{best}} - \mu(x)}{\sigma(x)} \ll -3$, both $\Phi(z) \to 0$ and $\phi(z) \to 0$, rendering $\nabla_x \text{EI}(x) \approx 0$. Gradient descent trajectories initialized in these flat plateaus stall immediately.
2. **Extreme Multimodality and Spurious Local Optima:**
   The EI surface contains numerous narrow spikes near boundary edges and between clusters of observations. Multi-start L-BFGS-B routinely gets trapped in unpromising local maxima unless hundreds of restarts are allocated.
3. **Multi-Objective Sequential Overhead:**
   In batched MTBO, $M+1$ distinct acquisition surfaces must be optimized at every step. Running multi-start L-BFGS-B sequentially for $M+1$ objectives incurs substantial computational latency.

---

### 4.2 Quasi-Random Sobol Sequences: Theory and Advantages
A **Sobol sequence** is a low-discrepancy deterministic quasi-random sequence designed to fill a multidimensional hypercube $[0, 1]^d$ with maximal uniformity.

#### Mathematical Discrepancy Scaling
The star discrepancy $D_N^*(\mathcal{P})$ measures the deviation between the empirical point density and uniform Lebesgue measure:
- For uniform pseudorandom Monte Carlo:
  $$D_N^* = \mathcal{O}\left( \frac{\sqrt{\log \log N}}{\sqrt{N}} \right) \approx \mathcal{O}\left( N^{-0.5} \right)$$
- For a quasi-random Sobol sequence:
  $$D_N^* = \mathcal{O}\left( \frac{(\log N)^d}{N} \right) \approx \mathcal{O}\left( N^{-1} \right)$$

Sobol sequences avoid both the **gaps** (large empty voids) and **clusters** (redundant nearby points) characteristic of pseudorandom sampling.

```
Pseudorandom Sampling (Monte Carlo)        Quasi-Random Sobol Sequence
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│  •    •  •                      │        │  •       •       •       •      │
│         • •   •                 │        │      •       •       •       •  │
│                   •             │        │  •       •       •       •      │
│  •                              │        │      •       •       •       •  │
│        [Large Voids]    •       │        │  •       •       •       •      │
│   •  •  • [Clusters]            │        │      •       •       •       •  │
└─────────────────────────────────┘        └─────────────────────────────────┘
```

#### Why Sobol Pool Maximization is Sensible for MTBO:
1. **Fully Vectorized Tensor Evaluation:**
   Evaluating a candidate pool of $P = 1,024$ or $2,048$ Sobol points across all $M+1$ tasks does not require sequential iteration. The entire candidate pool is evaluated through the GP posterior in a **single batched linear algebra pass**, executing in milliseconds.
2. **Immunity to Vanishing Gradients:**
   Because Sobol candidates are drawn globally across $[0, 1]^d$, the pool unconditionally evaluates high-uncertainty regions regardless of gradient flatness.
3. **Hybrid Two-Stage Refinement:**
   Our architecture employs Sobol sampling as a global coarse solver to locate the global basin of attraction, followed by an L-BFGS-B gradient polish seeded directly at the best Sobol candidate on the posterior mean $\mu(x)$, combining global robustness with continuous precision.

---

## 5. Metrics for Validating Gaussian Process Surrogate Models

In standard machine learning, model validation relies on a held-out test split. In Bayesian Optimization, every evaluation is a costly simulation of coupled differential equations over 20,000 steps; setting aside 20% of evaluations as an idle test set is prohibitive.

Instead, we employ **analytical Leave-One-Out Cross-Validation (LOO-CV)**, **standardized uncertainty calibration**, and **adaptive anomaly tracking**, providing rigorous validation at zero additional simulation cost.

### 5.1 Analytical Leave-One-Out Cross-Validation (LOO-CV)
LOO-CV systematically evaluates how well the surrogate predicts unvisited points by hiding observation $i$ and evaluating the error using a model trained on the remaining $N-1$ points.

#### The Closed-Form Shortcut
In standard models (such as neural networks), LOO-CV requires retraining the model $N$ times. In Gaussian Processes, once the covariance matrix $\mathbf{K}_y = \mathbf{K} + \sigma_n^2 \mathbf{I}$ has been inverted, the leave-one-out predictive distribution $p(f_i \mid \mathbf{y}_{-i})$ is available in **exact closed form** via matrix partitioning identities:

$$\mu_{-i}(x_i) = y_i - \frac{[\mathbf{K}_y^{-1}\mathbf{y}]_i}{[\mathbf{K}_y^{-1}]_{ii}}$$

$$\sigma_{-i}^2(x_i) = \frac{1}{[\mathbf{K}_y^{-1}]_{ii}}$$

Evaluating LOO predictions for all $N$ observations requires only vector-vector products with the precomputed inverse $\mathbf{K}_y^{-1}$, executing in under 1 millisecond.

---

### 5.2 Leave-One-Out Root Mean Square Error ($\text{RMSE}_{\text{LOO}}$)
From the LOO prediction errors $e_i = y_i - \mu_{-i}(x_i)$, the Root Mean Square Error measures the absolute prediction accuracy on unseen points:

$$\boxed{\text{RMSE}_{\text{LOO}} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( y_i - \mu_{-i}(x_i) \right)^2}}$$

#### Benchmark Interpretation:
- **$\text{RMSE}_{\text{LOO}} < 0.15$ log-units:** **Excellent.** Predictive error is within $15\%$ of an order of magnitude; the surrogate surface is highly trustworthy.
- **$\text{RMSE}_{\text{LOO}} \in [0.15, 0.40]$ log-units:** **Acceptable.** The model captures macro-trends and relative rankings, though fine local curvature has minor variance.
- **$\text{RMSE}_{\text{LOO}} > 0.80$ log-units:** **Poor.** The surrogate is failing to capture the underlying response surface.

---

### 5.3 Standardized Mean Squared Error ($\text{SMSE}$)
While RMSE evaluates absolute error, the **Standardized Mean Squared Error (SMSE)** normalizes the mean squared LOO error by the sample variance of the observations $\sigma_y^2 = \frac{1}{N-1}\sum_{i=1}^N (y_i - \bar{y})^2$:

$$\boxed{\text{SMSE} \triangleq \frac{\frac{1}{N} \sum_{i=1}^N \left( y_i - \mu_{-i}(x_i) \right)^2}{\sigma_y^2} = \frac{\text{MSE}_{\text{LOO}}}{\text{Var}(\mathbf{y})}}$$

#### Benchmark Interpretation:
- $\text{SMSE} = 1.0$: The model is no better than predicting the constant sample mean $\bar{y}$ (zero predictive skill).
- $\text{SMSE} \to 0.0$: Perfect predictive accuracy.
- The quantity $R^2_{\text{LOO}} \triangleq 1 - \text{SMSE}$ quantifies the **fraction of total variance explained** by the Gaussian Process on unseen data. In our Experiment 06 benchmarks:
  - `ZOGrandientPlay` ($K=5$ SAA): $\text{SMSE} = 0.0086 \implies \mathbf{99.1\%}$ of regret variance explained.
  - `NESeeking5D`: $\text{SMSE} = 0.0148 \implies \mathbf{98.5\%}$ of regret variance explained.

---

### 5.4 Uncertainty Calibration Metric ($\text{Std}(z)$)
Bayesian Optimization depends fundamentally on its predictive variance $\sigma(x)$ to balance exploration and exploitation. If $\sigma(x)$ is inaccurate:
- **Overconfidence ($\sigma$ too small):** The model ceases exploring and gets trapped in sub-optimal local basins.
- **Underconfidence ($\sigma$ too large):** The model wanders aimlessly, treating well-explored spaces as uncharted.

To validate uncertainty estimates, we compute the **standardized residuals ($z$-scores)**:

$$z_i \triangleq \frac{y_i - \mu_{-i}(x_i)}{\sigma_{-i}(x_i)}$$

If the Gaussian Process uncertainty estimates are statistically well-calibrated, the set of standardized residuals $\{z_1, \dots, z_N\}$ must behave as independent draws from a standard Gaussian distribution:
$$z_i \sim \mathcal{N}(0, 1)$$

We compute the sample standard deviation of the residuals:

$$\boxed{\text{Std}(z) = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (z_i - \bar{z})^2}}$$

#### Calibration Regimes:
```text
           Underconfident                    Ideal Calibration                   Overconfident
     (Explores Excessively)             (Correct Exploration/Exploit)        (Trapped in Local Minima)
◄───────────────────────────────┼────────────────────────────────────────┼───────────────────────────────►
        Std(z) < 0.70                     0.80 ≤ Std(z) ≤ 1.25                     Std(z) > 1.50
```
- **$\text{Std}(z) \in [0.80, 1.25]$:** **Ideal Calibration.** Empirical errors match the predicted epistemic standard deviations. When the model reports a $95\%$ confidence interval ($\pm 1.96\sigma$), unseen ground-truth values fall inside that interval exactly $95\%$ of the time.
- **$\text{Std}(z) > 1.50$:** **Severe Overconfidence.** Real prediction errors are much larger than the model's reported $\sigma$, caused by underestimated observation noise or realization variance in stochastic dynamics.
- **$\text{Std}(z) < 0.70$:** **Underconfidence.** The model overestimates uncertainty, slowing down convergence.

In our overnight benchmarks:
- `ZOGrandientPlay` with single simulations yielded $\text{Std}(z) = 1.5269$ (overconfident due to single-run realization noise).
- Introducing Sample Average Approximation ($K = 5$ SAA with Common Random Numbers) successfully brought $\text{Std}(z)$ down to **$1.0926$**, establishing near-ideal Gaussian calibration.
- `NESeeking5D` naturally achieved an almost perfect calibration of **$\text{Std}(z) = 0.9945$** due to its deterministic dynamical trajectory.

---

### 5.5 Adaptive Anomaly Detection and Refinement
Because Bayesian optimization relies on surrogate approximations, an **anomaly** can occur:
$$L(\theta_{\text{default}}^*; G_j) < L(\theta_j^*; G_j) \implies \Delta_j < 0$$
This occurs when the global consensus default $\theta_{\text{default}}^*$ outperforms the tailored candidate $\theta_j^*$ on Game $j$, violating theoretical optimality.

#### Root Causes of Anomalies:
1. **Surrogate Approximation Error:** The GP selected $\theta_j^*$ based on an inaccurate surrogate minimum $\mu_j(x)$ that suffered from high residual error.
2. **Negative Multi-Task Transfer:** The ICM task covariance $\mathbf{K}_T$ regularized the posterior of Game $j$ toward the ensemble mean, biasing it away from its true idiosyncratic optimum.
3. **Realization Noise (The Optimizer's Curse):** In stochastic algorithms (`ZOGrandientPlay`), a parameter can appear optimal due to a single "lucky" noise trajectory, failing when re-evaluated.

#### The Post-Evaluation Refinement Mechanism
To resolve anomalies without corrupting the objective function, the architecture executes a three-way verification:
1. Continuous surrogate gradient polish on the candidate default.
2. Continuous surrogate gradient polish on the candidate tailored optimum.
3. Single parallel verification batch evaluating all candidates under the true simulator.
If an anomaly is detected ($\Delta_j < 0$), the tailored parameter is updated to match the default:
$$\theta_j^* \longleftarrow \theta_{\text{default}}^*$$
guaranteeing by construction that the empirical tunability index is non-negative:
$$\Delta_j = \max\left(0, \; L(\theta_{\text{default}}^*; G_j) - L(\theta_j^*; G_j)\right) \ge 0$$

---

## 6. Summary Matrix: Technical Concepts and Mathematical Formulations

| Concept | Mathematical Formulation | Primary Purpose in Pipeline |
| :--- | :--- | :--- |
| **GP Prior** | $f(x) \sim \mathcal{GP}(m(x), k(x, x'))$ | Non-parametric functional prior over black-box regret surfaces. |
| **Gaussian Conditioning** | $\mu_* = \mathbf{k}_*^T \mathbf{K}_y^{-1}\mathbf{y}, \; \sigma_*^2 = k_{**} - \mathbf{k}_*^T \mathbf{K}_y^{-1}\mathbf{k}_*$ | Closed-form Bayesian posterior updates from simulation data. |
| **Intrinsic Coregionalization (ICM)** | $\mathbf{K}_{\text{joint}} = \mathbf{K}_T \otimes \mathbf{K}_X$ | Learns cross-game correlations via low-rank latent processes $\mathbf{W}\mathbf{W}^T$. |
| **Expected Improvement (EI)** | $(y_{\text{best}} - \mu)\Phi(z) + \sigma\phi(z)$ | Balances exploitation ($\mu$) and exploration ($\sigma$) in closed form. |
| **Tunability Index** | $\Delta_j = L(\theta_{\text{default}}^*; G_j) - L(\theta_j^*; G_j)$ | Quantifies the performance gain of game-tailored tuning over a default. |
| **Sobol Pool Maximization** | $x_{N+1} = \arg\max_{x \in \mathcal{P}_{\text{Sobol}}} \text{EI}(x)$ | Vectorized acquisition optimization avoiding gradient vanishing. |
| **LOO RMSE** | $\sqrt{\frac{1}{N} \sum (y_i - \mu_{-i})^2}$ | Measures out-of-sample surrogate predictive error without a holdout set. |
| **Standardized MSE (SMSE)** | $\text{MSE}_{\text{LOO}} / \text{Var}(\mathbf{y})$ | Measures the fraction of unexplained variance ($1 - \text{SMSE} = R^2_{\text{LOO}}$). |
| **Uncertainty Calibration** | $\text{Std}\left(\frac{y_i - \mu_{-i}}{\sigma_{-i}}\right) \approx 1.0$ | Verifies that epistemic uncertainty matches empirical errors. |
| **Sample Average Approx. (SAA)** | $\hat{L}_K(\theta) = \frac{1}{K}\sum_{k=1}^K L(\theta; \omega_k)$ | Suppresses realization noise by $\sqrt{K}$, repairing GP calibration. |
