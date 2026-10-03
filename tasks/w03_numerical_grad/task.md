# Central-difference gradient checker

Implement two functions:

- `numerical_grad(f, x, eps=1e-6)`: the central-difference gradient of a scalar function `f` at array `x`
  (any shape). Do not modify `x` in place permanently.
- `rel_error(a, b)`: `max |a - b| / max(1e-12, |a| + |b|)` element-wise, returned as a float.
