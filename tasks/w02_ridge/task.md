# Ridge regression in closed form

Implement `ridge_fit(X, y, lam, fit_intercept=True)` returning `(w, b)` that minimizes
`||y - X w - b||^2 + lam * ||w||^2` (the intercept `b` is **not** penalized). Use the closed form you derived;
solve the linear system rather than forming an explicit inverse. With `fit_intercept=False`, return `b = 0.0`.
