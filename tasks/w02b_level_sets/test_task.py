import numpy as np
from submission import *  # noqa: F401,F403
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
