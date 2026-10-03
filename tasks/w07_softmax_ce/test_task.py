import numpy as np
from submission import *  # noqa: F401,F403
def test_uniform():
    loss, d = softmax_cross_entropy(np.zeros((3, 4)), np.array([0, 1, 2]))
    assert abs(loss - np.log(4)) < 1e-12
    assert np.allclose(d.sum(1), 0)

def test_gradient_numeric():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 3)); y = rng.integers(0, 3, 5)
    _, d = softmax_cross_entropy(z, y)
    num = np.zeros_like(z)
    for i in range(5):
        for j in range(3):
            e = np.zeros_like(z); e[i, j] = 1e-6
            num[i, j] = (softmax_cross_entropy(z + e, y)[0] - softmax_cross_entropy(z - e, y)[0]) / 2e-6
    assert np.allclose(d, num, atol=1e-7)

def test_large_logits():
    z = np.array([[1e4, 0.0, -1e4]]); loss, d = softmax_cross_entropy(z, np.array([1]))
    assert np.isfinite(loss) and abs(loss - 1e4) < 1e-6 and np.all(np.isfinite(d))
