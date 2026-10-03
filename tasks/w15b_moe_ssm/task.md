# Mixture-of-experts routing and a discretized state-space layer

Two building blocks of modern architectures.

**Mixture of experts.**
- `topk_route(logits, k)`: logits (T, E) → `(idx, gates)`: the indices (T, k) of the k largest logits per token
  (largest first) and gates = softmax over those k logits only.
- `load_balance_loss(logits, idx)`: the Switch-Transformer auxiliary loss `E · Σ_i f_i · P_i`, where f_i is the
  fraction of tokens whose **top-1** expert is i and P_i is the mean (over tokens) softmax probability of expert i.

**State-space model** (single input/output channel, state size N): x' = A x + B u, y = C x.
- `discretize_zoh(A, B, dt)`: zero-order hold → `(Ad, Bd)` with Ad = exp(A·dt), Bd = A⁻¹(Ad − I)B.
- `ssm_recurrent(Ad, Bd, C, u)`: x_0 = 0; x_t = Ad x_{t−1} + Bd u_t; y_t = C x_t. Return y (L,).
- `ssm_kernel(Ad, Bd, C, L)`: K_j = C Ad^j Bd for j = 0…L−1.
- `ssm_convolutional(Ad, Bd, C, u)`: y = causal convolution of u with the kernel (must equal the recurrence).
