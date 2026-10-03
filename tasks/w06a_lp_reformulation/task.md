# Reformulating as linear programs: LAD regression and an L∞ fit

Use `scipy.optimize.linprog` (method="highs").

- `lad_regression(X, y)`: minimize Σ|yᵢ − xᵢᵀw| by introducing tᵢ ≥ |yᵢ − xᵢᵀw| (two linear inequalities each);
  return w.
- `chebyshev_regression(X, y)`: minimize max_i |yᵢ − xᵢᵀw| with one epigraph variable s; return `(w, s)`.
- `diet_problem(cost, nutrients, minimum)`: choose non-negative amounts x of each food to minimize costᵀx subject to
  nutrients @ x ≥ minimum (nutrients has one row per nutrient, one column per food); return `(x, total_cost)`.
