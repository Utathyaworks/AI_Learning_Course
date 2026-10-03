# Generalized advantage estimation

Implement `gae(rewards, values, dones, last_value, gamma, lam)` for one rollout of length T.

- `values[t]` is V(s_t); `last_value` is V(s_T) used to bootstrap after the final step.
- `dones[t] = True` means the episode ended **after** step t (no bootstrapping across it).
- `δ_t = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)`; `A_t = δ_t + γλ(1 − done_t)·A_{t+1}`.
Return `(advantages, returns)` with `returns = advantages + values`.
