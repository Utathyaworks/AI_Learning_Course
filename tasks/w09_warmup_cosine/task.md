# Linear warmup + cosine decay schedule

Implement `lr_at(step, total_steps, warmup_steps, base_lr, min_lr=0.0)` (steps start at 0).

- Warmup: for `step < warmup_steps`, `lr = base_lr * (step + 1) / warmup_steps`, so it reaches `base_lr`
  at `step = warmup_steps - 1`.
- Then cosine decay from `base_lr` at `step = warmup_steps` to `min_lr` at `step = total_steps - 1`.
- For `step ≥ total_steps`, return `min_lr`.
