"""Coding tasks, part 9 (Week 2B functions and graphs, Week 6A formulating optimization problems)."""

T = []

T.append(dict(
id="w02b_function_graphs", title="Reading functions numerically: transformations, roots, shape and power laws",
prompt="""Tools that check what a sketch claims, on a grid or by search (no calculus needed).

- `transform(f, a=1, b=1, c=0, d=0)`: return the function g(x) = a·f(b·(x − c)) + d.
- `bisect_root(f, lo, hi, tol=1e-10)`: a root of f in [lo, hi] by bisection (assume f(lo), f(hi) have opposite
  signs; raise ValueError otherwise).
- `is_monotone(y)`: `"increasing"`, `"decreasing"` or `"neither"` for samples y on a grid (non-strict: allow equal
  neighbours; a constant array counts as `"increasing"`).
- `is_convex_on_grid(y, tol=1e-12)`: True if every second difference y[i−1] − 2y[i] + y[i+1] ≥ −tol (equally spaced
  grid).
- `fit_power_law(x, y)`: fit y = a·x^k by a straight line on log-log axes; return `(a, k)`.
- `fit_exponential(x, y)`: fit y = a·e^(k·x) by a straight line on semilog-y axes; return `(a, k)`.""",
starter='''import numpy as np

def transform(f, a=1, b=1, c=0, d=0):
    raise NotImplementedError

def bisect_root(f, lo, hi, tol=1e-10):
    raise NotImplementedError

def is_monotone(y):
    raise NotImplementedError

def is_convex_on_grid(y, tol=1e-12):
    raise NotImplementedError

def fit_power_law(x, y):
    raise NotImplementedError

def fit_exponential(x, y):
    raise NotImplementedError
''',
solution='''import numpy as np

def transform(f, a=1, b=1, c=0, d=0):
    return lambda x: a * f(b * (np.asarray(x, float) - c)) + d

def bisect_root(f, lo, hi, tol=1e-10):
    flo, fhi = f(lo), f(hi)
    if flo == 0:
        return float(lo)
    if fhi == 0:
        return float(hi)
    if np.sign(flo) == np.sign(fhi):
        raise ValueError("f(lo) and f(hi) must have opposite signs")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi); fm = f(mid)
        if fm == 0:
            return float(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
    return float(0.5 * (lo + hi))

def is_monotone(y):
    d = np.diff(np.asarray(y, float))
    if np.all(d >= 0):
        return "increasing"
    if np.all(d <= 0):
        return "decreasing"
    return "neither"

def is_convex_on_grid(y, tol=1e-12):
    y = np.asarray(y, float)
    return bool(np.all(y[:-2] - 2 * y[1:-1] + y[2:] >= -tol))

def fit_power_law(x, y):
    k, loga = np.polyfit(np.log(x), np.log(y), 1)
    return float(np.exp(loga)), float(k)

def fit_exponential(x, y):
    k, loga = np.polyfit(np.asarray(x, float), np.log(y), 1)
    return float(np.exp(loga)), float(k)
''',
tests='''
def test_transform():
    g = transform(np.square, a=2, b=3, c=1, d=-4)
    x = np.array([0.0, 1.0, 2.5])
    assert np.allclose(g(x), 2 * (3 * (x - 1)) ** 2 - 4)
    assert np.isclose(transform(np.exp, c=2)(2.0), 1.0)          # shifted right by 2

def test_bisect():
    r = bisect_root(lambda x: x ** 3 - 2, 0, 2)
    assert abs(r - 2 ** (1 / 3)) < 1e-9
    assert abs(bisect_root(np.cos, 0, 3) - np.pi / 2) < 1e-9
    try:
        bisect_root(np.square, 1, 2); assert False
    except ValueError:
        pass

def test_shape_checks():
    x = np.linspace(-3, 3, 601)
    assert is_monotone(np.tanh(x)) == "increasing" and is_monotone(np.exp(-x)) == "decreasing"
    assert is_monotone(x ** 2) == "neither" and is_monotone(np.ones(5)) == "increasing"
    assert is_convex_on_grid(np.abs(x)) and is_convex_on_grid(np.log1p(np.exp(x))) and is_convex_on_grid(np.maximum(0, 1 - x))
    assert not is_convex_on_grid(np.sin(x)) and not is_convex_on_grid(-x ** 2)

def test_fits():
    x = np.linspace(1, 100, 50)
    a, k = fit_power_law(x, 3.0 * x ** -0.076)
    assert np.isclose(a, 3.0) and np.isclose(k, -0.076)
    a, k = fit_exponential(x / 10, 0.5 * np.exp(0.7 * x / 10))
    assert np.isclose(a, 0.5) and np.isclose(k, 0.7)
'''))

T.append(dict(
id="w02b_level_sets", title="Surfaces and level sets: grids, critical points and quadratic shapes",
prompt="""Two-variable functions without calculus.

- `grid_eval(f, xs, ys)`: evaluate f(x, y) on the grid; return Z of shape (len(ys), len(xs)) with Z[i, j] = f(xs[j], ys[i])
  (the `np.meshgrid` default convention).
- `grid_argmin(Z, xs, ys)`: the (x, y) location of the smallest value in Z.
- `quadratic_shape(A)`: for a symmetric 2×2 matrix, classify the surface z = [x y] A [x y]ᵀ as `"bowl"` (both
  eigenvalues > 0), `"cap"` (both < 0), `"saddle"` (opposite signs) or `"trough"` (one eigenvalue ≈ 0, |λ| < 1e-12).
- `level_set_ellipse_axes(A, c)`: for a positive-definite A, the contour xᵀAx = c is an ellipse; return its semi-axis
  lengths sorted from long to short (√(c/λ) for each eigenvalue λ).
- `contour_crossings(z_line, level)`: indices i where the sampled values z_line cross `level` between i and i+1
  (strict sign change of z − level).""",
starter='''import numpy as np

def grid_eval(f, xs, ys):
    raise NotImplementedError

def grid_argmin(Z, xs, ys):
    raise NotImplementedError

def quadratic_shape(A):
    raise NotImplementedError

def level_set_ellipse_axes(A, c):
    raise NotImplementedError

def contour_crossings(z_line, level):
    raise NotImplementedError
''',
solution='''import numpy as np

def grid_eval(f, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return f(X, Y)

def grid_argmin(Z, xs, ys):
    i, j = np.unravel_index(np.argmin(Z), Z.shape)
    return float(xs[j]), float(ys[i])

def quadratic_shape(A):
    lam = np.linalg.eigvalsh(np.asarray(A, float))
    if np.any(np.abs(lam) < 1e-12):
        return "trough"
    if np.all(lam > 0):
        return "bowl"
    if np.all(lam < 0):
        return "cap"
    return "saddle"

def level_set_ellipse_axes(A, c):
    lam = np.linalg.eigvalsh(np.asarray(A, float))
    return np.sort(np.sqrt(c / lam))[::-1]

def contour_crossings(z_line, level):
    s = np.sign(np.asarray(z_line, float) - level)
    return np.nonzero(s[:-1] * s[1:] < 0)[0]
''',
tests='''
def test_grid_eval_and_argmin():
    xs, ys = np.linspace(-2, 2, 81), np.linspace(-1, 3, 81)
    Z = grid_eval(lambda x, y: (x - 1) ** 2 + 100 * (y - x ** 2) ** 2, xs, ys)   # Rosenbrock
    assert Z.shape == (81, 81) and np.isclose(Z[0, 0], (xs[0] - 1) ** 2 + 100 * (ys[0] - xs[0] ** 2) ** 2)
    assert np.allclose(grid_argmin(Z, xs, ys), (1.0, 1.0))

def test_shapes():
    assert quadratic_shape([[2, 0], [0, 1]]) == "bowl"
    assert quadratic_shape([[-1, 0], [0, -3]]) == "cap"
    assert quadratic_shape([[1, 0], [0, -1]]) == "saddle"
    assert quadratic_shape([[1, 1], [1, 1]]) == "trough"
    assert quadratic_shape([[1, 2], [2, 1]]) == "saddle"     # off-diagonal terms matter

def test_ellipse_axes():
    assert np.allclose(level_set_ellipse_axes([[4, 0], [0, 1]], 4), [2, 1])
    R = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
    A = R @ np.diag([9.0, 1.0]) @ R.T
    assert np.allclose(level_set_ellipse_axes(A, 9), [3, 1])

def test_crossings():
    x = np.linspace(0, 2 * np.pi, 200)
    assert len(contour_crossings(np.sin(x), 0.5)) == 2
    assert len(contour_crossings(np.ones(10), 0.5)) == 0
'''))

T.append(dict(
id="w06a_formulations", title="From formulation to solution: proximal and projected gradient",
prompt="""Solve formulated problems with first-order methods.

- `soft_threshold(v, t)`: the proximal operator of t·‖x‖₁ (sign(v)·max(|v| − t, 0)).
- `lasso_ista(X, y, lam, iters=5000)`: minimize ½‖Xw − y‖² + lam·‖w‖₁ by ISTA with step 1/L, L = largest
  eigenvalue of XᵀX, starting from w = 0. Return w.
- `project_l2_ball(v, r)`: Euclidean projection onto {x : ‖x‖ ≤ r}.
- `project_simplex(v)`: Euclidean projection onto {x : x ≥ 0, Σx = 1} (sort-based algorithm).
- `ridge_constrained(X, y, r, iters=5000)`: minimize ½‖Xw − y‖² subject to ‖w‖ ≤ r by projected gradient
  (step 1/L, start at 0).
- `ridge_penalized(X, y, lam)`: closed-form argmin of ½‖Xw − y‖² + (lam/2)‖w‖².""",
starter='''import numpy as np

def soft_threshold(v, t):
    raise NotImplementedError

def lasso_ista(X, y, lam, iters=5000):
    raise NotImplementedError

def project_l2_ball(v, r):
    raise NotImplementedError

def project_simplex(v):
    raise NotImplementedError

def ridge_constrained(X, y, r, iters=5000):
    raise NotImplementedError

def ridge_penalized(X, y, lam):
    raise NotImplementedError
''',
solution='''import numpy as np

def soft_threshold(v, t):
    v = np.asarray(v, float)
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)

def _L(X):
    return float(np.linalg.eigvalsh(X.T @ X).max())

def lasso_ista(X, y, lam, iters=5000):
    L = _L(X); w = np.zeros(X.shape[1])
    for _ in range(iters):
        w = soft_threshold(w - X.T @ (X @ w - y) / L, lam / L)
    return w

def project_l2_ball(v, r):
    v = np.asarray(v, float); n = np.linalg.norm(v)
    return v if n <= r else v * (r / n)

def project_simplex(v):
    v = np.asarray(v, float)
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    k = np.nonzero(u - (css - 1) / np.arange(1, len(v) + 1) > 0)[0][-1]
    tau = (css[k] - 1) / (k + 1)
    return np.maximum(v - tau, 0.0)

def ridge_constrained(X, y, r, iters=5000):
    L = _L(X); w = np.zeros(X.shape[1])
    for _ in range(iters):
        w = project_l2_ball(w - X.T @ (X @ w - y) / L, r)
    return w

def ridge_penalized(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
''',
tests='''
from scipy.optimize import minimize

def _data(seed=0, n=60, d=8):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d)); w = np.zeros(d); w[:3] = [2.0, -1.5, 1.0]
    return X, X @ w + 0.1 * r.normal(size=n)

def test_soft_threshold():
    assert np.allclose(soft_threshold([-3, -0.5, 0, 0.5, 3], 1), [-2, 0, 0, 0, 2])

def test_lasso_is_sparse_and_optimal():
    X, y = _data()
    lam = 5.0
    w = lasso_ista(X, y, lam)
    assert np.sum(np.abs(w) > 1e-8) <= 4 and np.all(np.abs(w[:3]) > 0.5)
    g = X.T @ (X @ w - y)                      # KKT: |g_j| <= lam where w_j = 0, g_j = -lam sign(w_j) otherwise
    nz = np.abs(w) > 1e-8
    assert np.allclose(g[nz], -lam * np.sign(w[nz]), atol=1e-5) and np.all(np.abs(g[~nz]) <= lam + 1e-5)

def test_projections():
    v = np.array([3.0, 4.0])
    assert np.allclose(project_l2_ball(v, 1), [0.6, 0.8]) and np.allclose(project_l2_ball(v, 10), v)
    p = project_simplex([0.5, 2.0, -1.0])
    assert np.allclose(p, [0, 1, 0]) and np.isclose(project_simplex([0.2, 0.3, 0.1]).sum(), 1)
    r = np.random.default_rng(1)
    for _ in range(20):
        v = r.normal(size=6) * 3; p = project_simplex(v)
        res = minimize(lambda x: np.sum((x - v) ** 2), np.ones(6) / 6, constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                       bounds=[(0, None)] * 6, method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
        assert np.allclose(p, res.x, atol=1e-5)

def test_penalized_and_constrained_ridge_agree():
    X, y = _data(2)
    lam = 20.0
    wp = ridge_penalized(X, y, lam)
    wc = ridge_constrained(X, y, np.linalg.norm(wp), iters=20000)
    assert np.allclose(wc, wp, atol=1e-6)        # same point for the matching radius
'''))

T.append(dict(
id="w06a_lp_reformulation", title="Reformulating as linear programs: LAD regression and an L∞ fit",
prompt="""Use `scipy.optimize.linprog` (method="highs").

- `lad_regression(X, y)`: minimize Σ|yᵢ − xᵢᵀw| by introducing tᵢ ≥ |yᵢ − xᵢᵀw| (two linear inequalities each);
  return w.
- `chebyshev_regression(X, y)`: minimize max_i |yᵢ − xᵢᵀw| with one epigraph variable s; return `(w, s)`.
- `diet_problem(cost, nutrients, minimum)`: choose non-negative amounts x of each food to minimize costᵀx subject to
  nutrients @ x ≥ minimum (nutrients has one row per nutrient, one column per food); return `(x, total_cost)`.""",
starter='''import numpy as np
from scipy.optimize import linprog

def lad_regression(X, y):
    raise NotImplementedError

def chebyshev_regression(X, y):
    raise NotImplementedError

def diet_problem(cost, nutrients, minimum):
    raise NotImplementedError
''',
solution='''import numpy as np
from scipy.optimize import linprog

def lad_regression(X, y):
    n, d = X.shape
    c = np.concatenate([np.zeros(d), np.ones(n)])
    A = np.block([[X, -np.eye(n)], [-X, -np.eye(n)]])
    b = np.concatenate([y, -y])
    bounds = [(None, None)] * d + [(0, None)] * n
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    return r.x[:d]

def chebyshev_regression(X, y):
    n, d = X.shape
    c = np.concatenate([np.zeros(d), [1.0]])
    ones = np.ones((n, 1))
    A = np.block([[X, -ones], [-X, -ones]])
    b = np.concatenate([y, -y])
    r = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * d + [(0, None)], method="highs")
    return r.x[:d], float(r.x[d])

def diet_problem(cost, nutrients, minimum):
    r = linprog(np.asarray(cost, float), A_ub=-np.asarray(nutrients, float), b_ub=-np.asarray(minimum, float),
                bounds=[(0, None)] * len(cost), method="highs")
    return r.x, float(r.fun)
''',
tests='''
from scipy.optimize import minimize

def test_lad_resists_outliers():
    r = np.random.default_rng(0)
    x = np.linspace(0, 10, 40); X = np.column_stack([np.ones_like(x), x])
    y = 1.0 + 2.0 * x + 0.05 * r.normal(size=40); y[::8] += 30          # 5 big outliers
    w = lad_regression(X, y)
    ls = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(w, [1, 2], atol=0.1) and abs(ls[0] - 1) > 1
    obj = lambda v: np.abs(y - X @ v).sum()
    assert obj(w) <= obj(ls) + 1e-9 and obj(w) <= minimize(obj, ls, method="Nelder-Mead").fun + 1e-6

def test_chebyshev():
    X = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])
    w, s = chebyshev_regression(X, np.array([0.0, 2.0, 0.0]))
    assert np.isclose(s, 1.0) and np.allclose(w, [1.0, 0.0])
    assert np.isclose(np.max(np.abs(np.array([0, 2, 0]) - X @ w)), s)

def test_diet():
    cost = [2.0, 3.0]; nutrients = [[1, 2], [3, 1]]; minimum = [8, 9]
    x, total = diet_problem(cost, nutrients, minimum)
    assert np.isclose(total, 13.0) and np.allclose(x, [2, 3]) and np.all(np.array(nutrients) @ x >= np.array(minimum) - 1e-9)
'''))
