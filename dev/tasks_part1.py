"""Coding tasks, part 1 (Weeks 1–6 and C1). Built into tasks/ by dev/build_tasks.py."""

T = []

T.append(dict(
id="w01_gram_schmidt", title="Gram–Schmidt orthonormalization",
prompt="""Implement classical Gram–Schmidt (with re-orthogonalization optional) for the columns of `A`.

- `A` has shape (m, n) with linearly independent columns, m ≥ n.
- Return `Q` of shape (m, n) with orthonormal columns spanning the same space, column by column,
  so that `A = Q R` for some upper-triangular `R` with positive diagonal.
- Do not call `np.linalg.qr`.""",
starter='''import numpy as np

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    """Return Q (m, n) with orthonormal columns, same span as A, positive R diagonal."""
    raise NotImplementedError
''',
solution='''import numpy as np

def gram_schmidt(A):
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for _ in range(2):  # twice for numerical stability
            v -= Q[:, :j] @ (Q[:, :j].T @ v)
        Q[:, j] = v / np.linalg.norm(v)
    return Q
''',
tests='''
def test_orthonormal():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 5))
    Q = gram_schmidt(A)
    assert Q.shape == (8, 5)
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-8)

def test_same_span_and_positive_r():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 4))
    Q = gram_schmidt(A)
    R = Q.T @ A
    assert np.allclose(np.tril(R, -1), 0, atol=1e-8)
    assert np.all(np.diag(R) > 0)
    assert np.allclose(Q @ R, A, atol=1e-8)

def test_matches_qr_up_to_sign():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    Q = gram_schmidt(A)
    Q2, R2 = np.linalg.qr(A)
    Q2 = Q2 * np.sign(np.diag(R2))
    assert np.allclose(Q, Q2, atol=1e-8)
'''))

T.append(dict(
id="w01_broadcast_shape", title="NumPy broadcasting rule",
prompt="""Implement NumPy's broadcasting rule yourself.

`broadcast_shape(*shapes)` takes any number of shape tuples and returns the broadcast result shape.
If the shapes are incompatible, raise `ValueError`. Do not call any NumPy broadcasting helper.""",
starter='''def broadcast_shape(*shapes: tuple) -> tuple:
    """Return the broadcast shape of all input shapes, or raise ValueError."""
    raise NotImplementedError
''',
solution='''def broadcast_shape(*shapes):
    n = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(1, n + 1):
        dims = {s[-i] for s in shapes if len(s) >= i} - {1}
        if len(dims) > 1:
            raise ValueError("incompatible shapes")
        out.append(dims.pop() if dims else 1)
    return tuple(reversed(out))
''',
tests='''
import pytest

def test_classic():
    assert broadcast_shape((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)

def test_scalar_and_many():
    assert broadcast_shape((), (3, 1), (1, 4)) == (3, 4)

def test_random_agrees_with_numpy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(1, 4)
        shapes = [tuple(int(x) for x in rng.choice([1, 2, 3], size=rng.integers(0, 4))) for _ in range(k)]
        try:
            expected = np.broadcast_shapes(*shapes)
        except ValueError:
            with pytest.raises(ValueError):
                broadcast_shape(*shapes)
        else:
            assert broadcast_shape(*shapes) == expected

def test_incompatible():
    with pytest.raises(ValueError):
        broadcast_shape((3,), (4,))

def test_zero_size():
    assert broadcast_shape((0, 1), (1, 5)) == (0, 5)
'''))

T.append(dict(
id="w02_lu_solve", title="Solve Ax = b with LU and partial pivoting",
prompt="""Implement `lu_solve(A, b)`: factor `PA = LU` with partial pivoting, then solve by forward and back
substitution. `A` is (n, n) and invertible; `b` is (n,) or (n, k). Return `x` with the same trailing shape as `b`.
Do not call `np.linalg.solve`, `np.linalg.inv` or SciPy.""",
starter='''import numpy as np

def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    raise NotImplementedError
''',
solution='''import numpy as np

def lu_solve(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    vec = b.ndim == 1
    B = b.reshape(len(b), -1).copy()
    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    perm = np.arange(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]
            perm[[k, p]] = perm[[p, k]]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    B = B[perm]
    y = np.zeros_like(B)
    for i in range(n):
        y[i] = B[i] - L[i, :i] @ y[:i]
    x = np.zeros_like(y)
    for i in reversed(range(n)):
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x[:, 0] if vec else x
''',
tests='''
def test_random_vector():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)); b = rng.normal(size=6)
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b), atol=1e-8)

def test_needs_pivoting():
    A = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b))

def test_matrix_rhs():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); B = rng.normal(size=(5, 3))
    x = lu_solve(A, B)
    assert x.shape == (5, 3)
    assert np.allclose(A @ x, B, atol=1e-8)

def test_small_pivot_stability():
    A = np.array([[1e-12, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b), atol=1e-8)
'''))

T.append(dict(
id="w02_ridge", title="Ridge regression in closed form",
prompt="""Implement `ridge_fit(X, y, lam, fit_intercept=True)` returning `(w, b)` that minimizes
`||y - X w - b||^2 + lam * ||w||^2` (the intercept `b` is **not** penalized). Use the closed form you derived;
solve the linear system rather than forming an explicit inverse. With `fit_intercept=False`, return `b = 0.0`.""",
starter='''import numpy as np

def ridge_fit(X, y, lam: float, fit_intercept: bool = True):
    """Return (w, b)."""
    raise NotImplementedError
''',
solution='''import numpy as np

def ridge_fit(X, y, lam, fit_intercept=True):
    X = np.asarray(X, float); y = np.asarray(y, float)
    if fit_intercept:
        xm, ym = X.mean(0), y.mean()
        Xc, yc = X - xm, y - ym
    else:
        Xc, yc = X, y
    d = X.shape[1]
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ yc)
    b = float(ym - xm @ w) if fit_intercept else 0.0
    return w, b
''',
tests='''
def _obj(X, y, w, b, lam):
    r = y - X @ w - b
    return r @ r + lam * w @ w

def test_matches_ols_when_lam_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)); y = X @ np.array([1.0, -2.0, 0.5]) + 3.0
    w, b = ridge_fit(X, y, 0.0)
    assert np.allclose(w, [1.0, -2.0, 0.5], atol=1e-8) and abs(b - 3.0) < 1e-8

def test_optimality():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 4)); y = rng.normal(size=40)
    w, b = ridge_fit(X, y, 2.5)
    base = _obj(X, y, w, b, 2.5)
    for _ in range(20):
        dw = rng.normal(size=4) * 1e-3; db = rng.normal() * 1e-3
        assert _obj(X, y, w + dw, b + db, 2.5) >= base - 1e-9

def test_no_intercept():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 2)); y = rng.normal(size=30)
    w, b = ridge_fit(X, y, 1.0, fit_intercept=False)
    assert b == 0.0
    assert np.allclose(w, np.linalg.solve(X.T @ X + np.eye(2), X.T @ y))

def test_shrinks_with_lambda():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 3)); y = rng.normal(size=30)
    n1 = np.linalg.norm(ridge_fit(X, y, 0.1)[0]); n2 = np.linalg.norm(ridge_fit(X, y, 100.0)[0])
    assert n2 < n1
'''))

T.append(dict(
id="w03_numerical_grad", title="Central-difference gradient checker",
prompt="""Implement two functions:

- `numerical_grad(f, x, eps=1e-6)`: the central-difference gradient of a scalar function `f` at array `x`
  (any shape). Do not modify `x` in place permanently.
- `rel_error(a, b)`: `max |a - b| / max(1e-12, |a| + |b|)` element-wise, returned as a float.""",
starter='''import numpy as np

def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    raise NotImplementedError

def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def numerical_grad(f, x, eps=1e-6):
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps; fp = f(x)
        x[i] = old - eps; fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g

def rel_error(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))))
''',
tests='''
def test_quadratic():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    f = lambda x: 0.5 * x @ A @ x
    x = np.array([1.0, -2.0])
    assert rel_error(numerical_grad(f, x), A @ x) < 1e-7

def test_matrix_input():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(3, 4)); v = rng.normal(size=4)
    f = lambda W: np.sum(np.tanh(W @ v))
    analytic = np.outer(1 - np.tanh(W @ v) ** 2, v)
    assert rel_error(numerical_grad(f, W), analytic) < 1e-6

def test_input_unchanged():
    x = np.array([1.0, 2.0, 3.0]); x0 = x.copy()
    numerical_grad(lambda z: np.sum(z ** 3), x)
    assert np.array_equal(x, x0)

def test_rel_error_values():
    assert rel_error(np.array([1.0]), np.array([1.0])) == 0.0
    assert abs(rel_error(np.array([1.0]), np.array([3.0])) - 0.5) < 1e-12
'''))

T.append(dict(
id="w03_scalar_autodiff", title="Scalar reverse-mode autodiff",
prompt="""Build a tiny reverse-mode autodiff `Value` class.

- `Value(data)` stores `.data` (float) and `.grad` (float, starts at 0).
- Support `+`, `*`, `-` (binary and unary), `**` with a numeric exponent, and `/`, including with plain numbers
  on either side (`2 * v`, `v + 1`, `1 / v`).
- Methods `tanh()`, `exp()`, `log()`.
- `backward()` on the output fills `.grad` of every node by reverse topological order. Gradients must
  accumulate when a node is used more than once.""",
starter='''import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
''',
solution='''import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @staticmethod
    def _wrap(o):
        return o if isinstance(o, Value) else Value(o)

    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _b():
            self.grad += out.grad; other.grad += out.grad
        out._backward = _b
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _b():
            self.grad += other.data * out.grad; other.grad += self.data * out.grad
        out._backward = _b
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), "**")
        def _b():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _b
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _b():
            self.grad += (1 - t * t) * out.grad
        out._backward = _b
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")
        def _b():
            self.grad += e * out.grad
        out._backward = _b
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def _b():
            self.grad += out.grad / self.data
        out._backward = _b
        return out

    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-self._wrap(o))
    def __rsub__(self, o): return self._wrap(o) - self
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o): return self * self._wrap(o) ** -1
    def __rtruediv__(self, o): return self._wrap(o) * self ** -1

    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
''',
tests='''
import math

def test_simple_chain():
    a = Value(2.0); b = Value(-3.0)
    c = a * b + a ** 2
    c.backward()
    assert abs(c.data - (-6 + 4)) < 1e-12
    assert abs(a.grad - (b.data + 2 * a.data)) < 1e-12
    assert abs(b.grad - a.data) < 1e-12

def test_reuse_accumulates():
    x = Value(3.0)
    y = x * x + x
    y.backward()
    assert abs(x.grad - 7.0) < 1e-12

def test_numbers_on_both_sides():
    x = Value(2.0)
    y = 1 / x + 2 * x - 3 + (5 - x) / 4
    y.backward()
    assert abs(y.data - (0.5 + 4 - 3 + 0.75)) < 1e-12
    assert abs(x.grad - (-0.25 + 2 - 0.25)) < 1e-12

def test_tanh_exp_log_against_numeric():
    def f(v):
        return ((v * 0.7).tanh() + v.exp() * 0.1 - (v + 2).log()) * v
    x = Value(0.4); y = f(x); y.backward()
    def g(t):
        return (math.tanh(0.7 * t) + 0.1 * math.exp(t) - math.log(t + 2)) * t
    num = (g(0.4 + 1e-6) - g(0.4 - 1e-6)) / 2e-6
    assert abs(x.grad - num) < 1e-6
'''))

T.append(dict(
id="w04_logistic_loss_grad", title="Stable logistic loss and gradient",
prompt="""Implement `logistic_loss_grad(X, y, w, b, l2=0.0)` for binary labels `y ∈ {0, 1}`.

Return `(loss, dw, db)` where `loss` is the **mean** negative log-likelihood plus `0.5 * l2 * ||w||^2`.
It must be numerically stable for very large |logits| (no overflow warnings, no `inf`/`nan`).""",
starter='''import numpy as np

def logistic_loss_grad(X, y, w, b: float, l2: float = 0.0):
    """Return (loss, dw, db)."""
    raise NotImplementedError
''',
solution='''import numpy as np

def logistic_loss_grad(X, y, w, b, l2=0.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    z = X @ w + b
    # log(1 + e^z) - y z, computed stably
    loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * w @ w
    p = np.exp(-np.logaddexp(0.0, -z))  # stable sigmoid
    r = (p - y) / len(y)
    return float(loss), X.T @ r + l2 * w, float(r.sum())
''',
tests='''
import warnings

def _num(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g

def test_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)); y = (rng.random(20) > 0.5).astype(float)
    w = rng.normal(size=3); b = 0.3
    loss, dw, db = logistic_loss_grad(X, y, w, b, l2=0.1)
    assert np.allclose(dw, _num(lambda v: logistic_loss_grad(X, y, v, b, 0.1)[0], w), atol=1e-6)
    assert abs(db - (logistic_loss_grad(X, y, w, b + 1e-6, 0.1)[0] - logistic_loss_grad(X, y, w, b - 1e-6, 0.1)[0]) / 2e-6) < 1e-6

def test_known_value():
    X = np.zeros((2, 1)); y = np.array([0.0, 1.0])
    loss, dw, db = logistic_loss_grad(X, y, np.zeros(1), 0.0)
    assert abs(loss - np.log(2)) < 1e-12 and abs(db) < 1e-12

def test_stable_for_huge_logits():
    X = np.array([[1000.0], [-1000.0]]); y = np.array([0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, dw, db = logistic_loss_grad(X, y, np.ones(1), 0.0)
    assert np.isfinite(loss) and abs(loss - 1000.0) < 1e-6
    assert np.all(np.isfinite(dw))
'''))

T.append(dict(
id="w04_roc_auc", title="ROC AUC from ranks",
prompt="""Implement `roc_auc(y_true, scores)` for binary labels without scikit-learn.

Use the rank (Mann–Whitney) formulation: AUC is the probability that a random positive scores higher than a random
negative, counting ties as one half. Handle ties correctly. Raise `ValueError` if only one class is present.""",
starter='''import numpy as np

def roc_auc(y_true, scores) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def roc_auc(y_true, scores):
    y = np.asarray(y_true).astype(bool); s = np.asarray(scores, float)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
''',
tests='''
import pytest

def _brute(y, s):
    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]
    tot = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return tot / (len(pos) * len(neg))

def test_perfect_and_reversed():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0

def test_ties():
    assert abs(roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) - 0.5) < 1e-12

def test_random_vs_brute():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, 30); y[0], y[1] = 0, 1
        s = rng.integers(0, 6, 30).astype(float)
        assert abs(roc_auc(y, s) - _brute(y, s)) < 1e-12

def test_one_class():
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.2, 0.4])
'''))

T.append(dict(
id="w05_pca", title="PCA from the covariance eigendecomposition",
prompt="""Implement `pca(X, k)` returning `(components, explained_variance, X_proj)`.

- Center `X` (n, d) by its column means.
- `components`: (k, d), rows are unit principal directions, ordered by decreasing variance.
- `explained_variance`: (k,), the variance along each direction (use the `n - 1` denominator).
- `X_proj`: (n, k), the centered data projected onto the components.
Use an eigendecomposition or SVD you call yourself; don't use scikit-learn.""",
starter='''import numpy as np

def pca(X: np.ndarray, k: int):
    """Return (components (k,d), explained_variance (k,), X_proj (n,k))."""
    raise NotImplementedError
''',
solution='''import numpy as np

def pca(X, k):
    X = np.asarray(X, float)
    Xc = X - X.mean(0)
    C = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(vals)[::-1][:k]
    comps = vecs[:, idx].T
    return comps, vals[idx], Xc @ comps.T
''',
tests='''
def test_shapes_and_orthonormal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    c, v, P = pca(X, 3)
    assert c.shape == (3, 5) and v.shape == (3,) and P.shape == (100, 3)
    assert np.allclose(c @ c.T, np.eye(3), atol=1e-8)
    assert np.all(np.diff(v) <= 1e-12)

def test_recovers_direction():
    rng = np.random.default_rng(1)
    t = rng.normal(size=500) * 5
    d = np.array([3.0, 4.0]) / 5
    X = np.outer(t, d) + rng.normal(size=(500, 2)) * 0.1 + 7
    c, v, P = pca(X, 1)
    assert abs(abs(c[0] @ d) - 1) < 1e-3

def test_variance_matches_projection():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4))
    c, v, P = pca(X, 4)
    assert np.allclose(P.var(0, ddof=1), v, atol=1e-8)
    assert np.allclose(v.sum(), np.trace(np.cov(X.T)), atol=1e-8)
'''))

T.append(dict(
id="w05_kmeans_step", title="One k-means (Lloyd) iteration",
prompt="""Implement `kmeans_step(X, centers)` returning `(labels, new_centers, inertia)`.

- `labels[i]`: index of the nearest center (squared Euclidean; ties go to the lower index).
- `new_centers[j]`: mean of points assigned to j; if a cluster is empty, keep its old center.
- `inertia`: sum of squared distances of points to their **old** assigned center.
Vectorize the distance computation (no Python loop over points).""",
starter='''import numpy as np

def kmeans_step(X: np.ndarray, centers: np.ndarray):
    raise NotImplementedError
''',
solution='''import numpy as np

def kmeans_step(X, centers):
    X = np.asarray(X, float); C = np.asarray(centers, float)
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    d2 = np.maximum(d2, 0)
    labels = np.argmin(d2, axis=1)
    inertia = float(d2[np.arange(len(X)), labels].sum())
    new = C.copy()
    for j in range(len(C)):
        m = labels == j
        if m.any():
            new[j] = X[m].mean(0)
    return labels, new, inertia
''',
tests='''
def test_two_blobs():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    C = np.array([[1.0, 1.0], [9.0, 9.0]])
    lab, new, inert = kmeans_step(X, C)
    assert list(lab) == [0, 0, 1, 1]
    assert np.allclose(new, [[0.0, 0.5], [10.0, 10.5]])
    assert abs(inert - (2 + 1 + 2 + 5)) < 1e-9

def test_empty_cluster_kept():
    X = np.array([[0.0], [1.0]]); C = np.array([[0.5], [100.0]])
    lab, new, _ = kmeans_step(X, C)
    assert new[1, 0] == 100.0

def test_objective_never_increases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); C = X[:4].copy()
    prev = np.inf
    for _ in range(10):
        lab, C, inert = kmeans_step(X, C)
        assert inert <= prev + 1e-9
        prev = inert
'''))

T.append(dict(
id="w06_adam_step", title="Adam and AdamW update",
prompt="""Implement one optimizer step: `adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=True)`.

Return `(w_new, m_new, v_new)`. `t` is the step number starting at 1; apply bias correction.
- `decoupled=True` is **AdamW**: `w ← w − lr·weight_decay·w` applied separately from the Adam update.
- `decoupled=False` is **Adam + L2**: add `weight_decay·w` to the gradient before the moments.""",
starter='''import numpy as np

def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=True):
    raise NotImplementedError
''',
solution='''import numpy as np

def adam_step(w, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=True):
    w = np.asarray(w, float); g = np.asarray(g, float)
    if not decoupled and weight_decay:
        g = g + weight_decay * w
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
    step = lr * mh / (np.sqrt(vh) + eps)
    if decoupled and weight_decay:
        w = w - lr * weight_decay * w
    return w - step, m, v
''',
tests='''
def test_first_step_size_is_lr():
    w = np.array([1.0, -2.0]); g = np.array([0.3, -5.0])
    w1, m, v = adam_step(w, g, np.zeros(2), np.zeros(2), 1, lr=0.01)
    assert np.allclose(w - w1, 0.01 * np.sign(g), atol=1e-6)

def test_matches_reference_sequence():
    rng = np.random.default_rng(0)
    w = rng.normal(size=3); m = np.zeros(3); v = np.zeros(3)
    wr, mr, vr = w.copy(), np.zeros(3), np.zeros(3)
    for t in range(1, 6):
        g = rng.normal(size=3)
        w, m, v = adam_step(w, g, m, v, t, lr=0.1, weight_decay=0.01)
        wr = wr - 0.1 * 0.01 * wr
        mr = 0.9 * mr + 0.1 * g; vr = 0.999 * vr + 0.001 * g * g
        wr = wr - 0.1 * (mr / (1 - 0.9 ** t)) / (np.sqrt(vr / (1 - 0.999 ** t)) + 1e-8)
    assert np.allclose(w, wr) and np.allclose(m, mr) and np.allclose(v, vr)

def test_adam_l2_differs_from_adamw():
    w = np.array([5.0]); g = np.array([0.1])
    a, *_ = adam_step(w, g, np.zeros(1), np.zeros(1), 1, lr=0.1, weight_decay=0.5, decoupled=True)
    b, *_ = adam_step(w, g, np.zeros(1), np.zeros(1), 1, lr=0.1, weight_decay=0.5, decoupled=False)
    assert abs(a[0] - (5.0 - 0.1 * 0.5 * 5.0 - 0.1)) < 1e-6
    assert abs(b[0] - (5.0 - 0.1)) < 1e-6
'''))

T.append(dict(
id="w06_gini_split", title="Best decision-tree split by Gini impurity",
prompt="""Implement `best_split(x, y)` for one numeric feature `x` and integer class labels `y`.

Return `(threshold, weighted_gini)` for the split `x <= threshold` vs `x > threshold` that minimizes the
weighted Gini impurity of the two children. Thresholds are midpoints between consecutive distinct sorted values.
If all `x` are equal, return `(None, gini(y))`.""",
starter='''import numpy as np

def gini(y) -> float:
    raise NotImplementedError

def best_split(x, y):
    raise NotImplementedError
''',
solution='''import numpy as np

def gini(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(1 - (p * p).sum())

def best_split(x, y):
    x = np.asarray(x, float); y = np.asarray(y)
    vals = np.unique(x)
    if len(vals) == 1:
        return None, gini(y)
    best = (None, np.inf)
    for a, b in zip(vals[:-1], vals[1:]):
        t = (a + b) / 2
        L, R = y[x <= t], y[x > t]
        g = (len(L) * gini(L) + len(R) * gini(R)) / len(y)
        if g < best[1] - 1e-15:
            best = (t, g)
    return best[0], float(best[1])
''',
tests='''
def test_gini_values():
    assert gini([0, 0, 0]) == 0.0
    assert abs(gini([0, 1]) - 0.5) < 1e-12
    assert abs(gini([0, 1, 2]) - 2 / 3) < 1e-12

def test_perfect_split():
    t, g = best_split([1, 2, 3, 10, 11, 12], [0, 0, 0, 1, 1, 1])
    assert abs(t - 6.5) < 1e-12 and g == 0.0

def test_constant_feature():
    t, g = best_split([5, 5, 5, 5], [0, 1, 0, 1])
    assert t is None and abs(g - 0.5) < 1e-12

def test_against_brute():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 10, 40).astype(float); y = (x + rng.integers(0, 4, 40) > 6).astype(int)
    t, g = best_split(x, y)
    vals = np.unique(x)
    cands = [(a + b) / 2 for a, b in zip(vals[:-1], vals[1:])]
    best = min((len(y[x <= c]) * gini(y[x <= c]) + len(y[x > c]) * gini(y[x > c])) / len(y) for c in cands)
    assert abs(g - best) < 1e-12
'''))

T.append(dict(
id="c1_slurm_array", title="Slurm job-array sweep",
prompt="""You will run hyperparameter sweeps as Slurm job arrays. Implement two helpers.

1. `grid_point(grid, index)`: `grid` is a dict of name → list of values. Return the configuration (dict) at
   position `index` of the Cartesian product, iterating keys in the dict's order with the **last key changing
   fastest** (like `itertools.product`). Raise `IndexError` for an out-of-range index.
2. `sbatch_script(job_name, grid, command, gpus=1, hours=4, cpus=4, mem_gb=16)`: return the text of an sbatch
   script that
   - starts with `#!/bin/bash`;
   - has `#SBATCH` lines for `--job-name`, `--array=0-(N-1)` where N is the grid size, `--gres=gpu:<gpus>`,
     `--time=HH:00:00` (two-digit hours), `--cpus-per-task`, `--mem=<mem_gb>G` and
     `--output=logs/%x_%A_%a.out`;
   - ends with a line running `command` followed by ` --index $SLURM_ARRAY_TASK_ID`.""",
starter='''def grid_point(grid: dict, index: int) -> dict:
    raise NotImplementedError

def sbatch_script(job_name: str, grid: dict, command: str, gpus: int = 1, hours: int = 4,
                  cpus: int = 4, mem_gb: int = 16) -> str:
    raise NotImplementedError
''',
solution='''import math

def grid_point(grid, index):
    keys = list(grid)
    sizes = [len(grid[k]) for k in keys]
    total = math.prod(sizes)
    if not 0 <= index < total:
        raise IndexError(index)
    out = {}
    for k, n in zip(reversed(keys), reversed(sizes)):
        index, r = divmod(index, n)
        out[k] = grid[k][r]
    return {k: out[k] for k in keys}

def sbatch_script(job_name, grid, command, gpus=1, hours=4, cpus=4, mem_gb=16):
    n = math.prod(len(v) for v in grid.values())
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{n - 1}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --time={hours:02d}:00:00",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem_gb}G",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "",
        f"{command} --index $SLURM_ARRAY_TASK_ID",
    ]
    return "\\n".join(lines) + "\\n"
''',
tests='''
import itertools
import pytest

GRID = {"lr": [1e-3, 3e-4], "wd": [0.0, 0.01, 0.1], "seed": [0, 1]}

def test_matches_itertools_product():
    combos = list(itertools.product(*GRID.values()))
    for i, c in enumerate(combos):
        assert grid_point(GRID, i) == dict(zip(GRID, c))

def test_out_of_range():
    with pytest.raises(IndexError):
        grid_point(GRID, 12)

def test_script_directives():
    s = sbatch_script("sweep", GRID, "python train.py", gpus=2, hours=7, cpus=8, mem_gb=32)
    lines = s.strip().splitlines()
    assert lines[0] == "#!/bin/bash"
    for d in ["--job-name=sweep", "--array=0-11", "--gres=gpu:2", "--time=07:00:00",
              "--cpus-per-task=8", "--mem=32G", "--output=logs/%x_%A_%a.out"]:
        assert any(l.startswith("#SBATCH") and d in l for l in lines), d
    assert lines[-1] == "python train.py --index $SLURM_ARRAY_TASK_ID"
'''))
