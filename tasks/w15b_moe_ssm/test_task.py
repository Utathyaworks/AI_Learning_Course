import numpy as np
from submission import *  # noqa: F401,F403
def test_topk():
    lg = np.array([[1.0, 3.0, 2.0, 0.0], [0.0, 0.0, 5.0, 1.0]])
    idx, g = topk_route(lg, 2)
    assert idx.tolist() == [[1, 2], [2, 3]]
    assert np.allclose(g[0], np.exp([3, 2]) / np.exp([3, 2]).sum()) and np.allclose(g.sum(1), 1)

def test_balance_loss():
    E, T = 4, 400
    lg = np.zeros((T, E)); lg[np.arange(T), np.arange(T) % E] = 1.0
    idx, _ = topk_route(lg, 1)
    assert np.isclose(load_balance_loss(lg, idx), 1.0)
    lg2 = np.zeros((T, E)); lg2[:, 0] = 5.0
    idx2, _ = topk_route(lg2, 1)
    assert load_balance_loss(lg2, idx2) > 3.5

def _sys(seed=0, N=4):
    r = np.random.default_rng(seed)
    A = -np.diag(r.uniform(0.5, 2.0, N)) + 0.1 * r.normal(size=(N, N))
    return A, r.normal(size=(N, 1)), r.normal(size=(1, N))

def test_zoh_scalar():
    Ad, Bd = discretize_zoh([[-2.0]], [[1.0]], 0.1)
    assert np.isclose(Ad[0, 0], np.exp(-0.2)) and np.isclose(Bd[0, 0], (1 - np.exp(-0.2)) / 2)

def test_recurrent_equals_convolutional():
    A, B, C = _sys()
    Ad, Bd = discretize_zoh(A, B, 0.05)
    u = np.random.default_rng(1).normal(size=60)
    assert np.allclose(ssm_recurrent(Ad, Bd, C, u), ssm_convolutional(Ad, Bd, C, u))
    assert np.allclose(ssm_kernel(Ad, Bd, C, 3)[0], (C @ Bd).item())
