# Scalar reverse-mode autodiff

Build a tiny reverse-mode autodiff `Value` class.

- `Value(data)` stores `.data` (float) and `.grad` (float, starts at 0).
- Support `+`, `*`, `-` (binary and unary), `**` with a numeric exponent, and `/`, including with plain numbers
  on either side (`2 * v`, `v + 1`, `1 / v`).
- Methods `tanh()`, `exp()`, `log()`.
- `backward()` on the output fills `.grad` of every node by reverse topological order. Gradients must
  accumulate when a node is used more than once.
