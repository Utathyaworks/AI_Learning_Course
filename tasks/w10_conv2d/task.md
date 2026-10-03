# Conv2d forward with stride and padding (im2col)

Implement `conv2d(x, w, b, stride=1, padding=0)` (cross-correlation, like deep-learning libraries).

Shapes: `x` (N, C, H, W), `w` (F, C, KH, KW), `b` (F,). Output (N, F, H_out, W_out) with
`H_out = (H + 2·padding − KH) // stride + 1` (same for width). Use an im2col-style vectorized approach
rather than six nested loops.
