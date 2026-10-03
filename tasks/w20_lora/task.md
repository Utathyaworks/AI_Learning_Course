# LoRA forward, merge and parameter count

Implement LoRA for a linear layer `y = x W` with frozen `W` (d_in, d_out).

- `lora_forward(x, W, A, B, alpha)`: `A` is (d_in, r), `B` is (r, d_out); return `x W + (alpha / r) · x A B`.
- `lora_merge(W, A, B, alpha)`: return the merged weight so that `x @ merged` equals `lora_forward`.
- `lora_init(d_in, d_out, r, seed=0)`: return `(A, B)` with `A ~ N(0, 1/r)` (any seeded Gaussian) and
  **B all zeros**, so the adapted layer starts identical to the base layer.
- `lora_param_count(d_in, d_out, r)`: number of trainable parameters.
