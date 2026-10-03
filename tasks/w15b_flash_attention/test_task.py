import numpy as np
from submission import *  # noqa: F401,F403
def _qkv(n=37, m=37, d=8, dv=5, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)), r.normal(size=(m, d)), r.normal(size=(m, dv))

def test_attention_rows():
    Q, K, V = _qkv()
    out = attention(Q, K, np.eye(37)[:, :37])
    assert np.allclose(out.sum(1), 1)

def test_tiled_equals_naive():
    Q, K, V = _qkv(n=20, m=53)
    for b in (1, 7, 16, 64):
        assert np.allclose(tiled_attention(Q, K, V, b), attention(Q, K, V), atol=1e-10)

def test_tiled_is_stable():
    Q, K, V = _qkv(n=10, m=40)
    assert np.allclose(tiled_attention(Q * 300, K, V, 8), attention(Q * 300, K, V), atol=1e-8)

def test_kv_cache_equals_causal():
    Q, K, V = _qkv()
    assert np.allclose(decode_with_cache(Q, K, V), attention(Q, K, V, causal=True))

def test_gqa():
    K = np.arange(2 * 3 * 4.0).reshape(2, 3, 4)
    E = gqa_expand(K, 8)
    assert E.shape == (8, 3, 4) and np.array_equal(E[0], K[0]) and np.array_equal(E[3], K[0]) and np.array_equal(E[4], K[1])
