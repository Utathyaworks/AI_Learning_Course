import numpy as np
from submission import *  # noqa: F401,F403
def _setup(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(6, 4)); y = rng.integers(0, 3, 6)
    P = dict(W1=rng.normal(size=(4, 5)), b1=rng.normal(size=5) * 0.1, W2=rng.normal(size=(5, 3)), b2=rng.normal(size=3) * 0.1)
    return X, y, P

def test_shapes():
    X, y, P = _setup()
    loss, g = mlp_loss_grads(X, y, **P)
    for k in P:
        assert g[k].shape == P[k].shape

def test_numeric_gradients():
    X, y, P = _setup(1)
    loss, g = mlp_loss_grads(X, y, **P)
    for k in P:
        num = np.zeros_like(P[k])
        it = np.nditer(P[k], flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            Q = {kk: v.copy() for kk, v in P.items()}
            Q[k][i] += 1e-6; lp = mlp_loss_grads(X, y, **Q)[0]
            Q[k][i] -= 2e-6; lm = mlp_loss_grads(X, y, **Q)[0]
            num[i] = (lp - lm) / 2e-6
        assert np.allclose(g[k], num, atol=1e-6), k
