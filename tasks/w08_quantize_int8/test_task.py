import numpy as np
from submission import *  # noqa: F401,F403
def test_dtype_and_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8)) * 3
    q, s = quantize(x)
    assert q.dtype == np.int8 and q.min() >= -127 and q.max() <= 127
    assert np.isclose(np.max(np.abs(q)), 127)

def test_error_bound_per_tensor():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    q, s = quantize(x)
    assert np.all(np.abs(dequantize(q, s) - x) <= s / 2 + 1e-12)

def test_per_channel_better():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 64)) * np.array([[0.01], [0.1], [1.0], [10.0]])
    q1, s1 = quantize(x); q2, s2 = quantize(x, axis=1)
    assert np.asarray(s2).shape == (4, 1)
    e1 = np.abs(dequantize(q1, s1) - x)[0].mean(); e2 = np.abs(dequantize(q2, s2) - x)[0].mean()
    assert e2 < e1

def test_zero_tensor():
    q, s = quantize(np.zeros(5))
    assert np.all(q == 0) and np.all(dequantize(q, s) == 0)
