import numpy as np
from submission import *  # noqa: F401,F403
def _w(d, seed):
    r = np.random.default_rng(seed)
    return [r.normal(size=(d, d)) / np.sqrt(d) for _ in range(4)]

def _ref_mha(X, Y, Ws, h, mask):
    """Independent reference: one head at a time, row by row."""
    Wq, Wk, Wv, Wo = Ws
    d = X.shape[1]; dh = d // h
    Q, K, V = X @ Wq, Y @ Wk, Y @ Wv
    heads = []
    for k in range(h):
        sl = slice(k * dh, (k + 1) * dh)
        rows = []
        for i in range(len(X)):
            s = np.array([Q[i, sl] @ K[j, sl] / np.sqrt(dh) for j in range(len(Y))])
            if mask is not None:
                s = np.where(mask[i], s, -np.inf)
            p = np.exp(s - s.max()); p /= p.sum()
            rows.append(p @ V[:, sl])
        heads.append(np.array(rows))
    return np.concatenate(heads, 1) @ Wo

def test_additive():
    r = np.random.default_rng(0)
    q, K, V = r.normal(size=4), r.normal(size=(6, 4)), r.normal(size=(6, 3))
    Wq, Wk, v = r.normal(size=(5, 4)), r.normal(size=(5, 4)), r.normal(size=5)
    c, w = additive_attention(q, K, V, Wq, Wk, v)
    e = np.array([v @ np.tanh(Wq @ q + Wk @ k) for k in K]); p = np.exp(e - e.max()); p /= p.sum()
    assert np.allclose(w, p) and np.allclose(c, p @ V)

def test_masks():
    assert make_mask(4, "causal").sum() == 10
    w = make_mask(6, "window", window=2)
    assert w[5, 4] and w[5, 5] and not w[5, 3] and not w[2, 3]
    p = make_mask(5, "prefix", prefix=3)
    assert p[0, 2] and p[1, 2] and not p[3, 4] and p[4, 3] and not p[2, 3]

def test_mha_matches_reference():
    r = np.random.default_rng(1); d, h = 16, 4
    X = r.normal(size=(7, d)); Ws = _w(d, 2)
    for mask in (None, make_mask(7, "causal"), make_mask(7, "window", window=3)):
        assert np.allclose(multihead_attention(X, *Ws, h, mask=mask), _ref_mha(X, X, Ws, h, mask), atol=1e-8)

def test_cross_and_latent():
    r = np.random.default_rng(3); d, h = 8, 2
    X, Y, L = r.normal(size=(5, d)), r.normal(size=(30, d)), r.normal(size=(3, d))
    Ws = _w(d, 4)
    assert np.allclose(multihead_attention(X, *Ws, h, Y=Y), _ref_mha(X, Y, Ws, h, None), atol=1e-8)
    out = latent_cross_attention(L, Y, *Ws, h)
    assert out.shape == (3, d) and np.allclose(out, _ref_mha(L, Y, Ws, h, None), atol=1e-8)
