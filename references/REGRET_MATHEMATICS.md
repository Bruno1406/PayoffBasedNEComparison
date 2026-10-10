# Mathematical Foundations of Regret in Continuous Games and Payoff-Based NE Seeking

This document provides the theoretical and mathematical foundations for using **Total Regret** as an optimization metric in Multi-Task Bayesian Optimization (MTBO) for payoff-based Nash Equilibrium seeking algorithms (*Zero-Order Gradient Play* and *Extremum Seeking*).

---

## 1. General Game-Theoretic Framework

Consider an $N$-player continuous normal-form game defined by:
- A set of players $\mathcal{N} = \{1, 2, \dots, N\}$.
- Individual action spaces $\Omega_i \subset \mathbb{R}^{m_i}$, with the joint action space defined as the Cartesian product:
  $$\Omega = \prod_{i=1}^N \Omega_i \subset \mathbb{R}^M, \quad M = \sum_{i=1}^N m_i$$
- Payoff functions $J_i: \Omega \to \mathbb{R}$, where $J_i(u_i, u_{-i})$ denotes the profit or utility of player $i$ given their action $u_i \in \Omega_i$ and the joint action profile of their opponents $u_{-i} \in \Omega_{-i} = \prod_{j \ne i} \Omega_j$.

At each discrete iteration step $t \in \{0, 1, \dots, T\}$, players simultaneously execute the action profile:
$$u(t) = \left(u_1(t), u_2(t), \dots, u_N(t)\right) \in \Omega$$
and receive instantaneous payoffs $J_i(u(t))$.

---

## 2. Instantaneous and Cumulative Regret

### 2.1 Best-Response Mapping
Given the actions $u_{-i}(t)$ played by all other players at step $t$, the **best response** of player $i$ is defined as the set of actions maximizing their individual payoff:
$$u_i^*(u_{-i}(t)) = \arg\max_{u_i' \in \Omega_i} J_i(u_i', u_{-i}(t))$$

The maximum payoff player $i$ could have achieved against $u_{-i}(t)$ is:
$$J_i^*(u_{-i}(t)) = \max_{u_i' \in \Omega_i} J_i(u_i', u_{-i}(t))$$

### 2.2 Instantaneous Regret (Best-Response Deficit)
The **instantaneous regret** $R_i(t)$ of player $i$ at iteration $t$ measures the counterfactual utility lost by not playing their best response:
$$R_i(t) = \max_{u_i' \in \Omega_i} J_i(u_i', u_{-i}(t)) - J_i(u_i(t), u_{-i}(t))$$

By definition of the supremum, the instantaneous regret is strictly non-negative:
$$R_i(t) \ge 0 \quad \forall i \in \mathcal{N}, \quad \forall t \ge 0$$
with $R_i(t) = 0$ if and only if $u_i(t) \in u_i^*(u_{-i}(t))$.

### 2.3 Cumulative and Systemic Regret
Over a finite horizon of $T$ iterations, the cumulative regret accumulated by player $i$ is:
$$\text{Regret}_i(T) = \sum_{t=0}^T R_i(t)$$

The **Systemic Total Regret** across all players over the entire trajectory is defined as:
$$\text{Total Regret}(T) = \sum_{i=1}^N \text{Regret}_i(T) = \sum_{t=0}^T \sum_{i=1}^N R_i(t)$$

In average form, the Mean Systemic Regret per step is:
$$\overline{R}(T) = \frac{1}{T + 1} \sum_{t=0}^T \sum_{i=1}^N R_i(t)$$

---

## 3. Nash Equilibrium as the Zero-Regret State

A joint action profile $u^* = (u_1^*, \dots, u_N^*) \in \Omega$ constitutes a **Nash Equilibrium (NE)** if and only if no player can unilaterally deviate to increase their payoff:
$$\forall i \in \mathcal{N}, \quad J_i(u_i^*, u_{-i}^*) = \max_{u_i' \in \Omega_i} J_i(u_i', u_{-i}^*)$$

Consequently, evaluating the instantaneous regret at $u^*$ gives:
$$u^* \text{ is a Nash Equilibrium} \iff R_i(u^*) = 0 \quad \forall i \in \mathcal{N} \iff \sum_{i=1}^N R_i(u^*) = 0$$

### Connection to the Nikaido-Isoda Gap Function
In the theory of variational inequalities and continuous games, this summation corresponds to the **Nikaido-Isoda Gap Function** $\Psi: \Omega \to \mathbb{R}_{\ge 0}$:
$$\Psi(u) = \sum_{i=1}^N \left( \max_{u_i' \in \Omega_i} J_i(u_i', u_{-i}) - J_i(u_i, u_{-i}) \right)$$

Properties of $\Psi(u)$:
1. $\Psi(u) \ge 0$ for all $u \in \Omega$.
2. $\Psi(u) = 0$ if and only if $u$ is a Nash Equilibrium.
3. Minimizing $\Psi(u)$ or $\sum_{t=0}^T \Psi(u(t))$ provides a natural **Lyapunov-like potential function** that drives learning trajectories directly toward the Nash equilibrium without requiring prior knowledge of $u^*$.

---

## 4. Analytical Derivation for the Cournot Duopoly Game

In the 2-firm Cournot duopoly price game (Frihauf et al. 2012):
- Number of players: $N = 2$.
- Action sets (price choices): $\Omega_1 = \Omega_2 = [0, 300]$.
- Market demand parameters: $S_d = 100$, price sensitivity $p = 0.2$ (so $\frac{1}{p} = 5$).
- Marginal production costs: $m_1 = m_2 = 30$.

The payoff functions are:
$$J_1(u_1, u_2) = \left(S_d - \frac{u_1 - u_2}{p}\right)(u_1 - m_1) = \left(100 - 5(u_1 - u_2)\right)(u_1 - 30)$$
$$J_2(u_1, u_2) = \left(\frac{u_1 - u_2}{p}\right)(u_2 - m_2) = 5(u_1 - u_2)(u_2 - 30)$$

### 4.1 Player 1 Derivation
Expanding $J_1(u_1, u_2)$:
$$J_1(u_1, u_2) = (100 - 5u_1 + 5u_2)(u_1 - 30)$$
$$J_1(u_1, u_2) = -5 u_1^2 + (250 + 5 u_2) u_1 - (3000 + 150 u_2)$$

#### First and Second Order Conditions:
$$\frac{\partial J_1}{\partial u_1} = -10 u_1 + 250 + 5 u_2 = 0 \implies u_1^*(u_2) = 25.0 + 0.5 \cdot u_2$$
$$\frac{\partial^2 J_1}{\partial u_1^2} = -10 < 0 \quad (\text{strictly concave in } u_1)$$

#### Feasibility Check:
Since $u_2 \in [0, 300]$:
$$u_2 = 0 \implies u_1^* = 25.0 \in [0, 300]$$
$$u_2 = 300 \implies u_1^* = 175.0 \in [0, 300]$$
The unconstrained maximizer $u_1^*(u_2)$ always lies strictly in the interior of $[0, 300]$.

#### Completing the Square:
We express $J_1(u_1, u_2)$ in vertex form with respect to $u_1$:
$$J_1(u_1, u_2) = -5 \left[ u_1^2 - 2(25 + 0.5 u_2)u_1 + (25 + 0.5 u_2)^2 \right] + 5(25 + 0.5 u_2)^2 - (3000 + 150 u_2)$$
$$J_1(u_1, u_2) = -5 \left(u_1 - (25.0 + 0.5 u_2)\right)^2 + \underbrace{\left[5(625 + 25 u_2 + 0.25 u_2^2) - 3000 - 150 u_2\right]}_{J_1^*(u_2)}$$

Simplifying the optimal value:
$$J_1^*(u_2) = 3125 + 125 u_2 + 1.25 u_2^2 - 3000 - 150 u_2 = 1.25 u_2^2 - 25 u_2 + 125 = 1.25 (u_2 - 10)^2$$

Subtracting actual payoff from optimal payoff yields the exact regret:
$$R_1(u_1, u_2) = J_1^*(u_2) - J_1(u_1, u_2) = 5 \cdot \left(u_1 - (25.0 + 0.5 \cdot u_2)\right)^2$$

---

### 4.2 Player 2 Derivation
Expanding $J_2(u_1, u_2)$:
$$J_2(u_1, u_2) = 5(u_1 - u_2)(u_2 - 30) = -5 u_2^2 + (5 u_1 + 150) u_2 - 150 u_1$$

#### First and Second Order Conditions:
$$\frac{\partial J_2}{\partial u_2} = -10 u_2 + 5 u_1 + 150 = 0 \implies u_2^*(u_1) = 15.0 + 0.5 \cdot u_1$$
$$\frac{\partial^2 J_2}{\partial u_2^2} = -10 < 0 \quad (\text{strictly concave in } u_2)$$

#### Feasibility Check:
Since $u_1 \in [0, 300]$:
$$u_1 = 0 \implies u_2^* = 15.0 \in [0, 300]$$
$$u_1 = 300 \implies u_2^* = 165.0 \in [0, 300]$$
The unconstrained maximizer $u_2^*(u_1)$ always lies strictly in the interior of $[0, 300]$.

#### Completing the Square:
We express $J_2(u_1, u_2)$ in vertex form with respect to $u_2$:
$$J_2(u_1, u_2) = -5 \left[ u_2^2 - 2(15 + 0.5 u_1)u_2 + (15 + 0.5 u_1)^2 \right] + 5(15 + 0.5 u_1)^2 - 150 u_1$$
$$J_2(u_1, u_2) = -5 \left(u_2 - (15.0 + 0.5 u_1)\right)^2 + \underbrace{\left[5(225 + 15 u_1 + 0.25 u_1^2) - 150 u_1\right]}_{J_2^*(u_1)}$$

Simplifying the optimal value:
$$J_2^*(u_1) = 1125 + 75 u_1 + 1.25 u_1^2 - 150 u_1 = 1.25 u_1^2 - 75 u_1 + 1125 = 1.25 (u_1 - 30)^2$$

Subtracting actual payoff from optimal payoff yields the exact regret:
$$R_2(u_1, u_2) = J_2^*(u_1) - J_2(u_1, u_2) = 5 \cdot \left(u_2 - (15.0 + 0.5 \cdot u_1)\right)^2$$

---

### 4.3 Equilibrium Consistency Check
At the Nash equilibrium $(u_1^*, u_2^*)$, both players simultaneously play best responses:
$$\begin{cases}
u_1^* = 25.0 + 0.5 u_2^* \\
u_2^* = 15.0 + 0.5 u_1^*
\end{cases}$$

Substitute $u_2^*$ into $u_1^*$:
$$u_1^* = 25.0 + 0.5(15.0 + 0.5 u_1^*) = 32.5 + 0.25 u_1^* \implies 0.75 u_1^* = 32.5 \implies u_1^* = \frac{130}{3} \approx 43.3333$$
$$u_2^* = 15.0 + 0.5 \left(\frac{130}{3}\right) = 15.0 + \frac{65}{3} = \frac{110}{3} \approx 36.6667$$

Evaluating regrets at this point:
$$R_1(u_1^*, u_2^*) = 5 \cdot \left(\frac{130}{3} - \left(25.0 + 0.5 \cdot \frac{110}{3}\right)\right)^2 = 5 \cdot (0)^2 = 0$$
$$R_2(u_1^*, u_2^*) = 5 \cdot \left(\frac{110}{3} - \left(15.0 + 0.5 \cdot \frac{130}{3}\right)\right)^2 = 5 \cdot (0)^2 = 0$$

---

## 5. Generalization to Strongly Monotone Quadratic Games

For a general quadratic game with $N$ players, player $i$'s payoff function is:
$$J_i(u) = \frac{1}{2} u_i^T H_{ii} u_i + u_i^T \left( \sum_{j \ne i} H_{ij} u_j + d_i \right) + c_i(u_{-i})$$
where $H_{ii} \in \mathbb{R}^{m_i \times m_i}$ is negative definite ($H_{ii} \prec 0$), ensuring strict concavity.

### 5.1 Linear Best-Response Mapping
Setting the gradient with respect to $u_i$ to zero:
$$\nabla_{u_i} J_i(u) = H_{ii} u_i + \sum_{j \ne i} H_{ij} u_j + d_i = 0$$
$$u_i^*(u_{-i}) = -H_{ii}^{-1} \left( \sum_{j \ne i} H_{ij} u_j + d_i \right)$$

### 5.2 Quadratic Regret Metric
Applying a second-order Taylor expansion of $J_i(\cdot, u_{-i})$ around the optimal response $u_i^*(u_{-i})$:
$$J_i(u_i, u_{-i}) = J_i(u_i^*(u_{-i}), u_{-i}) + \underbrace{\nabla_{u_i} J_i(u_i^*, u_{-i})^T}_{= 0} (u_i - u_i^*) + \frac{1}{1} \cdot \frac{1}{2} (u_i - u_i^*)^T H_{ii} (u_i - u_i^*)$$

Rearranging gives the instantaneous regret:
$$R_i(u) = J_i(u_i^*(u_{-i}), u_{-i}) - J_i(u_i, u_{-i}) = \frac{1}{2} \left(u_i - u_i^*(u_{-i})\right)^T (-H_{ii}) \left(u_i - u_i^*(u_{-i})\right)$$

#### Interpretation:
- Instantaneous regret is a **Mahalanobis squared distance** between the player's action $u_i$ and their best-response manifold $u_i^*(u_{-i})$.
- The metric tensor is given by $-\frac{1}{2} H_{ii} \succ 0$.
- In our Cournot game, $H_{11} = H_{22} = -10$, so $-\frac{1}{2} H_{ii} = 5$, directly recovering $5(u_i - u_i^*)^2$.

---

## 6. Strategic Comparison: Regret Minimization vs. Distance to Equilibrium

| Attribute | Euclidean Distance to NE ($\|u(t) - u^*\|_2$) | Total Regret Minimization ($\sum_t \sum_i R_i(t)$) |
| :--- | :--- | :--- |
| **Oracle Knowledge** | **Requires knowing $u^*$ in advance.** Cannot be used in real-world scenarios where the equilibrium is unknown. | **Does not require knowing $u^*$.** Only relies on payoff evaluations and local best-response structure. |
| **Exploration Cost** | Treats all dither/exploration noise symmetrically, regardless of whether it loses small or large amounts of profit. | **Directly penalizes the economic cost of exploration.** Large dither perturbations that sacrifice profits incur high regret penalties. |
| **Transient Awareness** | An algorithm that rapidly oscillates around $u^*$ can appear good under average distance. | Penalizes large oscillations and non-optimal deviations throughout the entire learning path. |
| **Economic Interpretation** | Purely geometric coordinate measure in Euclidean space. | Direct measure of economic rationality, bounded suboptimality, and regret minimization in online learning. |

---

## 7. Vectorized Implementation for MTBO

In [`payoffnash/mtbo.py`](file:///home/bruno/Documents/PayoffBasedNEComparison/payoffnash/mtbo.py), the regret metric is implemented as a vectorized cost function over the simulation history:

```python
def total_regret_cost(game: QuadraticGame, history: EstimateHistory) -> float:
    """
    Computes total systemic regret across all players over the entire trajectory.
    
    Inputs:
      - game: QuadraticGame instance
      - history: EstimateHistory containing action_history of shape (T+1, 2)
    
    Returns:
      - float: Total cumulative regret sum_{t=0}^T (R1(t) + R2(t))
    """
    # Actions played at each time step
    u1 = history.action_history[:, 0]
    u2 = history.action_history[:, 1]
    
    # Best-response curves
    br1 = 25.0 + 0.5 * u2
    br2 = 15.0 + 0.5 * u1
    
    # Instantaneous regrets (guaranteed non-negative)
    regret1 = 5.0 * (u1 - br1) ** 2
    regret2 = 5.0 * (u2 - br2) ** 2
    
    return float(np.sum(regret1 + regret2))
```
