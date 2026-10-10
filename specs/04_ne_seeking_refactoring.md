# Specification 04: Refactoring of `NESeeking.estimate` Method and Optimization of `ZOGrandientPlay`

**Target Modules:** `payoffnash/ne_seeking.py`, `payoffnash/zo_gradient_play.py`  
**Target Classes:** `NESeeking`, `ZOGrandientPlay`  

---

## 1. Executive Summary & Motivation

The `NESeeking.estimate` method currently requires a fixed step count (`num_steps` or `num_iters`), preallocates static arrays of fixed size, and returns only the terminal Nash estimate $\hat{u}$ as the first element of its return tuple.

This specification refactors `NESeeking` and standardizes both `NESeeking` and `ZOGrandientPlay`:
1. **Unified Stopping Criteria:** Replace dual step arguments with three orthogonal termination conditions: `num_steps`, `time_limit`, and `max_rel_error`.
2. **Elimination of Computational Waste:**
   - **Time Check:** Only invoke `time.time()` if `time_limit != np.inf`.
   - **Error Norms:** Only compute norms, averages, and relative errors if `max_rel_error > 0.0`.
3. **Initial Buffer Allocation Bug Fix (`num_steps + 1`):** Preallocate buffers of size `num_steps + 1` so that storing the initial state at row 0 plus `num_steps` iteration updates never triggers premature geometric doubling on the final step.
4. **0-Indexed History with Initial State:** Index 0 records the initial state ($t=0$). Steps $1 \dots s$ record post-update trajectories. Total returned length is $s + 1$.
5. **Windowed Non-Overlapping Relative Error for NESeeking:** Evaluated strictly across non-overlapping blocks of $P = 5$ dither periods to prevent false triggers from sinusoidal dither ripple.
6. **Full State History Output:** Return `u_hat_history` (or `state_history` in ZO) of shape `(s + 1, N * m)` instead of only the terminal vector.
7. **Strict API Enforcement:** Reject legacy parameter aliases (`num_iters`, `N_iters`) with an explicit `TypeError`.

---

## 2. Interface Specification (`NESeeking` & `ZOGrandientPlay`)

### Method Signature
```python
def estimate(
    self,
    initial_actions: np.ndarray, # (named initial_u_hat in NESeeking)
    max_rel_error: Optional[float] = 0.0,
    num_steps: Optional[int] = np.inf,
    time_limit: Optional[float] = np.inf,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
```

### Input Validation & Legacy Parameter Rejection
```python
if "num_iters" in kwargs or "N_iters" in kwargs:
    raise TypeError("num_iters and N_iters are deprecated; use num_steps instead.")

if (max_rel_error is None or max_rel_error == 0.0) and num_steps == np.inf and time_limit == np.inf:
    raise ValueError("At least one stopping criterion must be specified: max_rel_error, num_steps, or time_limit.")
```

---

## 3. Mathematical Formulation of Relative Error

### A. For `NESeeking` (Windowed Averaging across $P = 5$ Periods)
Extremum seeking updates oscillate at dither frequencies $\omega_i$. The slowest dither frequency dictates the fundamental period:
$$\omega_{\min} = \min_{i} \omega_i, \quad T_{\max} = \frac{2\pi}{\omega_{\min}}$$
The number of discrete forward Euler steps per period is:
$$N_{\text{period}} = \max\left(1, \operatorname{round}\left(\frac{2\pi}{\omega_{\min} \Delta t}\right)\right)$$

Let the window length be $W = 5 \times N_{\text{period}}$ steps.  
Only when `max_rel_error > 0.0`:
Accumulate $\hat{u}$ over non-overlapping blocks of $W$ steps. For block $k \ge 1$:
$$\bar{u}_k = \frac{1}{W} \sum_{s = k W}^{(k+1)W - 1} \hat{u}_s$$
At the boundary of block $k$ (every $W$ steps):
$$\text{rel\_error} = \frac{\|\bar{u}_k - \bar{u}_{k-1}\|_2}{\|\bar{u}_{k-1}\|_2 + 10^{-12}}$$
If `rel_error <= max_rel_error`, terminate.

### B. For `ZOGrandientPlay` (Step-to-Step State Diff)
Only when `max_rel_error > 0.0`:
$$\text{diff\_norm} = \|x_t - x_{t-1}\|_2$$
$$\text{rel\_error} = \frac{\text{diff\_norm}}{\|x_{t-1}\|_2 + 10^{-12}}$$
If `max_rel_error == 0.0`, this norm computation is completely skipped.

---

## 4. Preallocation & Dynamic Resizing Architecture

### Initial Buffer Allocation
* Capacity:
  ```python
  history_length = int(num_steps + 1) if num_steps != np.inf else 10000
  ```
  *(Fixes the bug in ZOGrandientPlay where `history_length = num_steps` caused array doubling on the last step).*
* Buffer allocation:
  ```python
  state_history = np.empty((history_length, self.game.N * self.m), dtype=float)
  action_history = np.empty((history_length, self.game.N * self.m), dtype=float)
  payoff_history = np.empty((history_length, self.game.N), dtype=float)
  ```

### Initial State at Row 0 ($t = 0$)
Before entering the loop:
* In `NESeeking`:
  ```python
  u_hat_history[0, :] = u_hat.ravel()
  action_history[0, :] = flat_u0
  payoff_history[0, :] = eta
  ```
* In `ZOGrandientPlay`:
  ```python
  flat_state0 = state.ravel()
  state_history[0, :] = flat_state0
  action_history[0, :] = flat_state0
  payoff_history[0, :] = np.array([self.game.payoff_functions[i](*flat_state0) for i in range(self.game.N)], dtype=float)
  ```

### Geometric Doubling Loop
At step $s \ge 1$:
If $s \ge \text{history\_length}$:
1. Allocate new buffers of size `history_length * 2`.
2. Copy existing data: `new_buffer[:history_length, :] = old_buffer`.
3. Update `history_length = history_length * 2`.

---

## 5. Execution Loop & Optimized Stopping Logic

```python
s = 0  # number of completed steps
track_time = (time_limit != np.inf)
track_error = (max_rel_error is not None and max_rel_error > 0.0)

start_time = time.time() if track_time else 0.0
rel_error = np.inf

while True:
    if s >= num_steps:
        break
    if track_time and (time.time() - start_time >= time_limit):
        break
    if track_error and rel_error <= max_rel_error:
        break

    s += 1
    # 1. Resize buffer if s >= history_length
    # 2. Compute dynamic update step
    # 3. Record state_history[s, :], action_history[s, :], payoff_history[s, :]
    # 4. If track_error: compute relative error
```

---

## 6. Return Values

Both estimators return a 3-tuple sliced to `s + 1` rows:
```python
return state_history[:s + 1], action_history[:s + 1], payoff_history[:s + 1]
```
Row 0 represents $t = 0$ (initial conditions), and row $s$ represents the final converged state.

---

## 7. Additional Requirements

1. **`zo_gradient_play.py` Update:**
   Apply the `num_steps + 1` preallocation, `state_history[0, :]` initialization, conditional `time.time()` and norm computations, and legacy keyword rejection. Clean up `estimator_2p` call in `main`.
2. **`mtbo.py` Integration:**
   Ensure `evaluate_task` passes `num_steps=self.num_iters` and unpacks `action_history = res[1]`.
3. **Test Suite Verification:**
   Ensure `tests/test_game_ensemble.py` and `tests/test_mtbo.py` pass cleanly with all new signatures and return formats.
