import numpy as np
from submission import *  # noqa: F401,F403
def _attn(Q, K, V):
    s = Q @ K.T / np.sqrt(Q.shape[1]); s -= s.max(1, keepdims=True)
    w = np.exp(s); w /= w.sum(1, keepdims=True)
    return w @ V

def test_matches_concatenated_attention():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(4, 8)), rng.normal(size=(6, 8)), rng.normal(size=(6, 8))
    Pk, Pv = rng.normal(size=(3, 8)), rng.normal(size=(3, 8))
    out, pw = prefix_attention(Q, K, V, Pk, Pv)
    assert np.allclose(out, _attn(Q, np.vstack([Pk, K]), np.vstack([Pv, V])))
    assert pw.shape == (4,) and np.all((pw > 0) & (pw < 1))

def test_dominant_prefix():
    Q = np.ones((1, 4)); K = np.zeros((2, 4)); V = np.zeros((2, 4))
    Pk = np.full((1, 4), 20.0); Pv = np.full((1, 4), 7.0)
    out, pw = prefix_attention(Q, K, V, Pk, Pv)
    assert pw[0] > 0.999 and np.allclose(out, 7.0, atol=1e-3)
