# LSTM cell forward

Implement one LSTM step `lstm_cell(x, h, c, W, U, b)` with batch inputs.

Shapes: `x` (N, D), `h` and `c` (N, H), `W` (D, 4H), `U` (H, 4H), `b` (4H,). Pre-activations
`a = x W + h U + b` are split in the order **input, forget, cell (g), output** along the last axis.
`i = σ(a_i)`, `f = σ(a_f)`, `g = tanh(a_g)`, `o = σ(a_o)`, `c' = f⊙c + i⊙g`, `h' = o⊙tanh(c')`.
Return `(h_new, c_new)`.
