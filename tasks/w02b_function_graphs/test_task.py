import numpy as np
from submission import *  # noqa: F401,F403
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
