import numpy as np
from submission import *  # noqa: F401,F403
def test_norm_preserved():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 16))
    y = apply_rope(x, np.arange(10))
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))

def test_position_zero_identity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8))
    assert np.allclose(apply_rope(x, [0, 0, 0]), x)

def test_relative_property():
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=(1, 32)), rng.normal(size=(1, 32))
    s1 = apply_rope(q, [7]) @ apply_rope(k, [3]).T
    s2 = apply_rope(q, [104]) @ apply_rope(k, [100]).T
    assert np.allclose(s1, s2)

def test_first_pair_angle():
    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    y = apply_rope(x, [2.0])
    assert np.allclose(y[0, :2], [np.cos(2.0), np.sin(2.0)])
