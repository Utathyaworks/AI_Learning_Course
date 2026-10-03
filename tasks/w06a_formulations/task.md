# From formulation to solution: proximal and projected gradient

Solve formulated problems with first-order methods.

- `soft_threshold(v, t)`: the proximal operator of t·‖x‖₁ (sign(v)·max(|v| − t, 0)).
- `lasso_ista(X, y, lam, iters=5000)`: minimize ½‖Xw − y‖² + lam·‖w‖₁ by ISTA with step 1/L, L = largest
  eigenvalue of XᵀX, starting from w = 0. Return w.
- `project_l2_ball(v, r)`: Euclidean projection onto {x : ‖x‖ ≤ r}.
- `project_simplex(v)`: Euclidean projection onto {x : x ≥ 0, Σx = 1} (sort-based algorithm).
- `ridge_constrained(X, y, r, iters=5000)`: minimize ½‖Xw − y‖² subject to ‖w‖ ≤ r by projected gradient
  (step 1/L, start at 0).
- `ridge_penalized(X, y, lam)`: closed-form argmin of ½‖Xw − y‖² + (lam/2)‖w‖².
