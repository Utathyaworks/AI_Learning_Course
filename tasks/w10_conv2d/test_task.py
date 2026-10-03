import numpy as np
from submission import *  # noqa: F401,F403
def _naive(x, w, b, s, p):
    N, C, H, W = x.shape; F, _, KH, KW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho = (H + 2 * p - KH) // s + 1; Wo = (W + 2 * p - KW) // s + 1
    out = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    out[n, f, i, j] = (xp[n, :, i*s:i*s+KH, j*s:j*s+KW] * w[f]).sum() + b[f]
    return out

def test_against_naive():
    rng = np.random.default_rng(0)
    for s, p in [(1, 0), (1, 1), (2, 1), (2, 0)]:
        x = rng.normal(size=(2, 3, 7, 6)); w = rng.normal(size=(4, 3, 3, 3)); b = rng.normal(size=4)
        assert np.allclose(conv2d(x, w, b, s, p), _naive(x, w, b, s, p))

def test_identity_kernel():
    x = np.arange(16.0).reshape(1, 1, 4, 4)
    w = np.zeros((1, 1, 3, 3)); w[0, 0, 1, 1] = 1
    assert np.allclose(conv2d(x, w, np.zeros(1), 1, 1), x)
