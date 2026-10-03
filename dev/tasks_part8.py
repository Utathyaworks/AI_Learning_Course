"""Coding tasks, part 8 (Week 15A: attention mechanisms in depth)."""

T = []

T.append(dict(
id="w15a_attention_types", title="Attention types: additive, multi-head with masks, sliding window, cross and latent",
prompt="""Implement in NumPy (no batch dimension):

- `additive_attention(q, K, V, Wq, Wk, v)`: Bahdanau scores e_j = vᵀ tanh(Wq q + Wk k_j) for a single query q (d,),
  keys K (m, d); return `(context, weights)`.
- `make_mask(n, kind, window=None, prefix=None)`: boolean (n, n), True = may attend. `kind` is `"full"`,
  `"causal"`, `"window"` (causal and |i − j| < window) or `"prefix"` (positions < prefix see each other freely;
  later positions are causal).
- `multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None)`: self-attention on X (n, d); if `Y` is given,
  cross-attention where queries come from X and keys/values from Y. Split d into h heads, scaled dot-product per
  head, concatenate, project with Wo. Masked positions get −∞ before the softmax.
- `latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h)`: Perceiver-style — a small set of latents L (k, d) attends to a
  long input X (n, d); output (k, d). (It's cross-attention with queries from L.)""",
starter='''import numpy as np

def additive_attention(q, K, V, Wq, Wk, v):
    raise NotImplementedError

def make_mask(n, kind, window=None, prefix=None):
    raise NotImplementedError

def multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None):
    raise NotImplementedError

def latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h):
    raise NotImplementedError
''',
solution='''import numpy as np

def _softmax(S):
    S = S - S.max(-1, keepdims=True)
    P = np.exp(S)
    return P / P.sum(-1, keepdims=True)

def additive_attention(q, K, V, Wq, Wk, v):
    e = np.tanh(q @ Wq.T + K @ Wk.T) @ v
    w = _softmax(e)
    return w @ V, w

def make_mask(n, kind, window=None, prefix=None):
    i, j = np.indices((n, n))
    if kind == "full":
        return np.ones((n, n), bool)
    if kind == "causal":
        return j <= i
    if kind == "window":
        return (j <= i) & (i - j < window)
    if kind == "prefix":
        return (j <= i) | ((i < prefix) & (j < prefix))
    raise ValueError(kind)

def multihead_attention(X, Wq, Wk, Wv, Wo, h, mask=None, Y=None):
    Y = X if Y is None else Y
    n, d = X.shape; dh = d // h
    Q = (X @ Wq).reshape(n, h, dh).transpose(1, 0, 2)
    K = (Y @ Wk).reshape(len(Y), h, dh).transpose(1, 0, 2)
    V = (Y @ Wv).reshape(len(Y), h, dh).transpose(1, 0, 2)
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dh)
    if mask is not None:
        S = np.where(mask[None], S, -np.inf)
    O = (_softmax(S) @ V).transpose(1, 0, 2).reshape(n, d)
    return O @ Wo

def latent_cross_attention(L, X, Wq, Wk, Wv, Wo, h):
    return multihead_attention(L, Wq, Wk, Wv, Wo, h, Y=X)
''',
tests='''
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
'''))

T.append(dict(
id="w15a_linear_positions", title="Linear attention as an RNN, RoPE, ALiBi and KV-cache memory",
prompt="""Implement:

- `linear_attention_quadratic(Q, K, V, causal=True)`: with feature map φ(x) = elu(x) + 1, output row i is
  Σ_{j≤i} (φ(q_i)·φ(k_j)) v_j / Σ_{j≤i} φ(q_i)·φ(k_j) (all j if not causal).
- `linear_attention_recurrent(Q, K, V)`: the same causal output computed as an RNN with state S = Σ φ(k_j) v_jᵀ
  (d × dv) and normalizer z = Σ φ(k_j), updated one step at a time.
- `rope(X, positions, base=10000.0)`: rotary embedding on X (n, d), d even — rotate each pair (x_{2i}, x_{2i+1}) by
  angle positions · base^(−2i/d).
- `alibi_bias(n, n_heads)`: (n_heads, n, n) causal bias −m_h·(i − j) for j ≤ i and −∞ for j > i, with slopes
  m_h = 2^(−8(h+1)/n_heads) for h = 0 … n_heads−1.
- `kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2)`: memory to cache K and V for one sequence.""",
starter='''import numpy as np

def linear_attention_quadratic(Q, K, V, causal=True):
    raise NotImplementedError

def linear_attention_recurrent(Q, K, V):
    raise NotImplementedError

def rope(X, positions, base=10000.0):
    raise NotImplementedError

def alibi_bias(n, n_heads):
    raise NotImplementedError

def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2):
    raise NotImplementedError
''',
solution='''import numpy as np

def _phi(x):
    x = np.asarray(x, float)
    return np.where(x > 0, x + 1.0, np.exp(np.minimum(x, 0)))

def linear_attention_quadratic(Q, K, V, causal=True):
    A = _phi(Q) @ _phi(K).T
    if causal:
        A = np.tril(A)
    return (A @ V) / A.sum(1, keepdims=True)

def linear_attention_recurrent(Q, K, V):
    fq, fk = _phi(Q), _phi(K)
    S = np.zeros((Q.shape[1], V.shape[1])); z = np.zeros(Q.shape[1]); out = []
    for t in range(len(Q)):
        S += np.outer(fk[t], V[t]); z += fk[t]
        out.append(fq[t] @ S / (fq[t] @ z))
    return np.array(out)

def rope(X, positions, base=10000.0):
    X = np.asarray(X, float); n, d = X.shape
    theta = base ** (-np.arange(0, d, 2) / d)
    ang = np.asarray(positions, float)[:, None] * theta[None]
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(X)
    out[:, 0::2] = X[:, 0::2] * c - X[:, 1::2] * s
    out[:, 1::2] = X[:, 0::2] * s + X[:, 1::2] * c
    return out

def alibi_bias(n, n_heads):
    m = 2.0 ** (-8.0 * (np.arange(n_heads) + 1) / n_heads)
    i, j = np.indices((n, n))
    B = -m[:, None, None] * (i - j)[None].astype(float)
    return np.where((j <= i)[None], B, -np.inf)

def kv_cache_bytes(n_layers, n_kv_heads, d_head, seq_len, bytes_per=2):
    return int(2 * n_layers * n_kv_heads * d_head * seq_len * bytes_per)
''',
tests='''
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
'''))
