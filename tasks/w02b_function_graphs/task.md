# Reading functions numerically: transformations, roots, shape and power laws

Tools that check what a sketch claims, on a grid or by search (no calculus needed).

- `transform(f, a=1, b=1, c=0, d=0)`: return the function g(x) = a·f(b·(x − c)) + d.
- `bisect_root(f, lo, hi, tol=1e-10)`: a root of f in [lo, hi] by bisection (assume f(lo), f(hi) have opposite
  signs; raise ValueError otherwise).
- `is_monotone(y)`: `"increasing"`, `"decreasing"` or `"neither"` for samples y on a grid (non-strict: allow equal
  neighbours; a constant array counts as `"increasing"`).
- `is_convex_on_grid(y, tol=1e-12)`: True if every second difference y[i−1] − 2y[i] + y[i+1] ≥ −tol (equally spaced
  grid).
- `fit_power_law(x, y)`: fit y = a·x^k by a straight line on log-log axes; return `(a, k)`.
- `fit_exponential(x, y)`: fit y = a·e^(k·x) by a straight line on semilog-y axes; return `(a, k)`.
