import numpy as np
from submission import *  # noqa: F401,F403
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
