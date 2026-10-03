import numpy as np
from submission import *  # noqa: F401,F403
def test_forward_stats():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(64, 5))
    out, *_ = batchnorm_forward_backward(x, np.ones(5), np.zeros(5), np.zeros((64, 5)))
    assert np.allclose(out.mean(0), 0, atol=1e-10) and np.allclose(out.std(0), 1, atol=1e-3)

def test_backward_numeric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 3)); g = rng.normal(size=3); b = rng.normal(size=3); dout = rng.normal(size=(6, 3))
    _, dx, dg, db = batchnorm_forward_backward(x, g, b, dout)
    f = lambda x_, g_, b_: (batchnorm_forward_backward(x_, g_, b_, dout)[0] * dout).sum()
    num = np.zeros_like(x)
    for i in range(6):
        for j in range(3):
            e = np.zeros_like(x); e[i, j] = 1e-6
            num[i, j] = (f(x + e, g, b) - f(x - e, g, b)) / 2e-6
    assert np.allclose(dx, num, atol=1e-6)
    numg = np.array([(f(x, g + e, b) - f(x, g - e, b)) / 2e-6 for e in np.eye(3) * 1e-6])
    assert np.allclose(dg, numg, atol=1e-6) and np.allclose(db, dout.sum(0))
