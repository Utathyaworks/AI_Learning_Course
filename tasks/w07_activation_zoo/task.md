# Activation zoo: forward, gradient and numerical stability

Implement these element-wise activations on NumPy arrays, each with its derivative `<name>_grad(x)`:

`sigmoid`, `relu` (use 0 as the gradient at 0), `leaky_relu(x, a=0.01)`, `elu(x, a=1.0)`,
`selu` (λ = 1.0507009873554805, α = 1.6732632423543772), `gelu` (exact, x·Φ(x)),
`silu` (x·σ(x)), `softplus` (log(1 + eˣ)).

Also implement:
- `gelu_tanh(x)`: the tanh approximation 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
- `log_sigmoid(x)`: log σ(x).
- `swiglu(x, W, V, W2)`: the SwiGLU MLP `(silu(x @ W) * (x @ V)) @ W2`.

`sigmoid`, `softplus` and `log_sigmoid` must be finite and accurate at |x| = 1000 (no overflow warnings turned
into NaN or inf).
