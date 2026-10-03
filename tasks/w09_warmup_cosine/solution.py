import math

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step >= total_steps:
        return min_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(1, total_steps - 1 - warmup_steps)
    p = (step - warmup_steps) / span
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))
