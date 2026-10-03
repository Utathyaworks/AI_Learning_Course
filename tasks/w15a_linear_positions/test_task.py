import numpy as np
from submission import *  # noqa: F401,F403
def _qkv(n=25, d=6, dv=4, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(n, d)), r.normal(size=(n, d)), r.normal(size=(n, dv))

def test_linear_attention_rnn_equals_quadratic():
    Q, K, V = _qkv()
    assert np.allclose(linear_attention_recurrent(Q, K, V), linear_attention_quadratic(Q, K, V), atol=1e-10)

def test_linear_attention_is_a_weighted_average():
    Q, K, _ = _qkv()
    out = linear_attention_quadratic(Q, K, np.ones((25, 1)), causal=False)
    assert np.allclose(out, 1)
    first = linear_attention_quadratic(Q, K, np.arange(25.0)[:, None])[0]
    assert np.isclose(first[0], 0.0)   # causal: position 0 only sees itself

def test_rope_relative_property_and_norm():
    r = np.random.default_rng(1); d = 8
    q, k = r.normal(size=(1, d)), r.normal(size=(1, d))
    dot = lambda a, b: (rope(q, [a]) @ rope(k, [b]).T).item()
    assert np.isclose(dot(10, 3), dot(107, 100)) and np.isclose(dot(5, 5), (q @ k.T).item())
    X = r.normal(size=(6, d))
    assert np.allclose(np.linalg.norm(rope(X, np.arange(6)), axis=1), np.linalg.norm(X, axis=1))

def test_alibi():
    B = alibi_bias(5, 8)
    assert B.shape == (8, 5, 5) and np.isneginf(B[0, 1, 2]) and B[0, 4, 4] == 0
    assert np.isclose(B[0, 4, 1], -3 * 2 ** -1) and np.isclose(B[7, 4, 1], -3 * 2 ** -8)

def test_kv_cache():
    mha = kv_cache_bytes(32, 32, 128, 32768)
    assert mha == 2 * 32 * 32 * 128 * 32768 * 2 == 17179869184   # 16 GiB in fp16
    assert kv_cache_bytes(32, 8, 128, 32768) * 4 == mha            # GQA with 8 KV heads: 4x smaller
    assert kv_cache_bytes(32, 1, 128, 32768) * 32 == mha           # MQA
