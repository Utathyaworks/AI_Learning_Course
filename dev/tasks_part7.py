"""Coding tasks, part 7 (Week 15B modern architectures, Week 19B mechanistic interpretability)."""

T = []

T.append(dict(
id="w15b_flash_attention", title="Online-softmax (tiled) attention, KV cache and GQA",
prompt="""Single-head shapes: Q (n, d), K (m, d), V (m, dv).

- `attention(Q, K, V, causal=False)`: softmax(QKᵀ/√d)V; with `causal=True`, query i may attend to keys 0…i
  (assume n == m).
- `tiled_attention(Q, K, V, block=16)`: the same result, computed over key/value blocks with the **online softmax**
  (keep a running max m, running sum l and running output o per query; rescale when the max grows). Never build the
  full n × m score matrix — work one key block at a time.
- `decode_with_cache(Q, K, V)`: generate causally one position at a time, appending each step's k, v to a cache
  and attending only over the cache; return the (n, dv) outputs (must equal causal attention).
- `gqa_expand(K, n_q_heads)`: K of shape (n_kv_heads, m, d) → (n_q_heads, m, d), each KV head shared by
  n_q_heads / n_kv_heads consecutive query heads.""",
starter='''import numpy as np

def attention(Q, K, V, causal=False):
    raise NotImplementedError

def tiled_attention(Q, K, V, block=16):
    raise NotImplementedError

def decode_with_cache(Q, K, V):
    raise NotImplementedError

def gqa_expand(K, n_q_heads):
    raise NotImplementedError
''',
solution='''import numpy as np

def attention(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        S = np.where(np.tril(np.ones(S.shape, bool)), S, -np.inf)
    S = S - S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)
    return P @ V

def tiled_attention(Q, K, V, block=16):
    n, d = Q.shape
    m_run = np.full(n, -np.inf); l_run = np.zeros(n); o = np.zeros((n, V.shape[1]))
    for s in range(0, K.shape[0], block):
        S = Q @ K[s:s + block].T / np.sqrt(d)
        m_new = np.maximum(m_run, S.max(1))
        scale = np.exp(m_run - m_new)
        P = np.exp(S - m_new[:, None])
        l_run = l_run * scale + P.sum(1)
        o = o * scale[:, None] + P @ V[s:s + block]
        m_run = m_new
    return o / l_run[:, None]

def decode_with_cache(Q, K, V):
    kc, vc, out = [], [], []
    for t in range(Q.shape[0]):
        kc.append(K[t]); vc.append(V[t])
        out.append(attention(Q[t:t + 1], np.array(kc), np.array(vc))[0])
    return np.array(out)

def gqa_expand(K, n_q_heads):
    g = n_q_heads // K.shape[0]
    return np.repeat(K, g, axis=0)
''',
tests='''
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
'''))

T.append(dict(
id="w15b_moe_ssm", title="Mixture-of-experts routing and a discretized state-space layer",
prompt="""Two building blocks of modern architectures.

**Mixture of experts.**
- `topk_route(logits, k)`: logits (T, E) → `(idx, gates)`: the indices (T, k) of the k largest logits per token
  (largest first) and gates = softmax over those k logits only.
- `load_balance_loss(logits, idx)`: the Switch-Transformer auxiliary loss `E · Σ_i f_i · P_i`, where f_i is the
  fraction of tokens whose **top-1** expert is i and P_i is the mean (over tokens) softmax probability of expert i.

**State-space model** (single input/output channel, state size N): x' = A x + B u, y = C x.
- `discretize_zoh(A, B, dt)`: zero-order hold → `(Ad, Bd)` with Ad = exp(A·dt), Bd = A⁻¹(Ad − I)B.
- `ssm_recurrent(Ad, Bd, C, u)`: x_0 = 0; x_t = Ad x_{t−1} + Bd u_t; y_t = C x_t. Return y (L,).
- `ssm_kernel(Ad, Bd, C, L)`: K_j = C Ad^j Bd for j = 0…L−1.
- `ssm_convolutional(Ad, Bd, C, u)`: y = causal convolution of u with the kernel (must equal the recurrence).""",
starter='''import numpy as np

def topk_route(logits, k):
    raise NotImplementedError

def load_balance_loss(logits, idx):
    raise NotImplementedError

def discretize_zoh(A, B, dt):
    raise NotImplementedError

def ssm_recurrent(Ad, Bd, C, u):
    raise NotImplementedError

def ssm_kernel(Ad, Bd, C, L):
    raise NotImplementedError

def ssm_convolutional(Ad, Bd, C, u):
    raise NotImplementedError
''',
solution='''import numpy as np
from scipy.linalg import expm

def topk_route(logits, k):
    idx = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, idx, 1)
    g = np.exp(top - top.max(1, keepdims=True))
    return idx, g / g.sum(1, keepdims=True)

def load_balance_loss(logits, idx):
    T, E = logits.shape
    p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    f = np.bincount(idx[:, 0], minlength=E) / T
    return float(E * np.sum(f * p.mean(0)))

def discretize_zoh(A, B, dt):
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ad = expm(A * dt)
    Bd = np.linalg.solve(A, (Ad - np.eye(len(A))) @ B)
    return Ad, Bd

def ssm_recurrent(Ad, Bd, C, u):
    x = np.zeros(Ad.shape[0]); y = []
    for ut in u:
        x = Ad @ x + Bd.ravel() * ut
        y.append(float(np.asarray(C).ravel() @ x))
    return np.array(y)

def ssm_kernel(Ad, Bd, C, L):
    K, v = [], Bd.ravel().copy()
    c = np.asarray(C).ravel()
    for _ in range(L):
        K.append(c @ v); v = Ad @ v
    return np.array(K)

def ssm_convolutional(Ad, Bd, C, u):
    K = ssm_kernel(Ad, Bd, C, len(u))
    return np.array([np.dot(K[:t + 1][::-1], u[:t + 1]) for t in range(len(u))])
''',
tests='''
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
'''))

T.append(dict(
id="w19b_circuits_patching", title="Circuits in a 1-layer attention-only transformer, and activation patching",
prompt="""A tiny 1-layer, attention-only transformer with one head, written in NumPy:

`forward(tokens, W)` with `W = dict(E, P, WQ, WK, WV, WO, U)`:
x0 = E[tokens] + P[:n]; scores = (x0 WQ)(x0 WK)ᵀ/√d_head with a causal mask; A = softmax(scores);
head = A (x0 WV) WO; x1 = x0 + head; logits = x1 U. Return `(logits, cache)` where
`cache = {"x0": x0, "A": A, "head": head}`.

Implement:
- `forward(tokens, W, patch=None)`: if `patch` is a dict like `{"head": array}`, use that array as the head output
  instead of computing it (activation patching).
- `qk_circuit(W)` = E WQ (E WK)ᵀ (vocab × vocab, ignoring positions) and `ov_circuit(W)` = E WV WO U (vocab × vocab).
- `direct_logit_attribution(cache, W, pos, token)`: split the logit of `token` at position `pos` into the
  contribution of the embedding path (x0 U) and of the head (head U); return `(emb, head)` — they sum to the logit.
- `patching_effect(clean, corrupt, W, pos, token)`: run the corrupted input with the head output patched in from
  the clean run; return the fraction of the clean-vs-corrupt logit difference for `token` at `pos` that patching
  restores: (patched − corrupt) / (clean − corrupt).""",
starter='''import numpy as np

def forward(tokens, W, patch=None):
    raise NotImplementedError

def qk_circuit(W):
    raise NotImplementedError

def ov_circuit(W):
    raise NotImplementedError

def direct_logit_attribution(cache, W, pos, token):
    raise NotImplementedError

def patching_effect(clean, corrupt, W, pos, token):
    raise NotImplementedError
''',
solution='''import numpy as np

def forward(tokens, W, patch=None):
    tokens = np.asarray(tokens); n = len(tokens)
    x0 = W["E"][tokens] + W["P"][:n]
    q, k = x0 @ W["WQ"], x0 @ W["WK"]
    S = q @ k.T / np.sqrt(q.shape[1])
    S = np.where(np.tril(np.ones((n, n), bool)), S, -np.inf)
    S = S - S.max(1, keepdims=True); A = np.exp(S); A /= A.sum(1, keepdims=True)
    head = A @ (x0 @ W["WV"]) @ W["WO"]
    if patch is not None and "head" in patch:
        head = patch["head"]
    logits = (x0 + head) @ W["U"]
    return logits, {"x0": x0, "A": A, "head": head}

def qk_circuit(W):
    return (W["E"] @ W["WQ"]) @ (W["E"] @ W["WK"]).T

def ov_circuit(W):
    return W["E"] @ W["WV"] @ W["WO"] @ W["U"]

def direct_logit_attribution(cache, W, pos, token):
    return float(cache["x0"][pos] @ W["U"][:, token]), float(cache["head"][pos] @ W["U"][:, token])

def patching_effect(clean, corrupt, W, pos, token):
    lc, cc = forward(clean, W)
    lx, _ = forward(corrupt, W)
    lp, _ = forward(corrupt, W, patch={"head": cc["head"]})
    return float((lp[pos, token] - lx[pos, token]) / (lc[pos, token] - lx[pos, token]))
''',
tests='''
def _W(seed=0, V=11, d=16, dh=8, n=12):
    r = np.random.default_rng(seed)
    return dict(E=r.normal(size=(V, d)), P=0.1 * r.normal(size=(n, d)), WQ=r.normal(size=(d, dh)) / 4,
                WK=r.normal(size=(d, dh)) / 4, WV=r.normal(size=(d, dh)) / 4, WO=r.normal(size=(dh, d)) / 4,
                U=r.normal(size=(d, V)))

def test_forward_shapes_and_causality():
    W = _W(); t = np.array([1, 4, 2, 7, 3])
    lg, c = forward(t, W)
    assert lg.shape == (5, 11) and np.allclose(c["A"].sum(1), 1) and np.allclose(np.triu(c["A"], 1), 0)
    lg2, _ = forward(np.array([1, 4, 2, 7, 9]), W)
    assert np.allclose(lg[:4], lg2[:4])          # changing the last token can't change earlier positions

def test_circuits():
    W = _W()
    assert qk_circuit(W).shape == (11, 11) and ov_circuit(W).shape == (11, 11)
    assert np.allclose(ov_circuit(W)[3], W["E"][3] @ W["WV"] @ W["WO"] @ W["U"])

def test_dla_sums_to_logit():
    W = _W(); t = np.array([5, 1, 8, 2, 2, 6])
    lg, c = forward(t, W)
    e, h = direct_logit_attribution(c, W, 4, 7)
    assert np.isclose(e + h, lg[4, 7])

def test_patching():
    W = _W(1); clean = np.array([3, 5, 1, 9, 2]); corrupt = np.array([3, 5, 1, 9, 2])
    corrupt = corrupt.copy(); corrupt[1] = 7
    assert np.isclose(patching_effect(clean, corrupt, W, 4, 0), 1.0)   # the head is the only path that sees token 1 at pos 4
    lg, c = forward(clean, W)
    same, _ = forward(clean, W, patch={"head": c["head"]})
    assert np.allclose(same, lg)
'''))

T.append(dict(
id="w19b_sparse_autoencoder", title="Sparse autoencoder: forward pass, loss and evaluation metrics",
prompt="""A sparse autoencoder (SAE) for activations X (n, d) with m latents:

f = ReLU((X − b_dec) W_enc + b_enc), X̂ = f W_dec + b_dec, with W_enc (d, m), W_dec (m, d).

Implement:
- `normalize_decoder(W_dec)`: rescale each latent's decoder row to unit L2 norm.
- `sae_forward(X, W_enc, b_enc, W_dec, b_dec)`: return `(f, X_hat)`.
- `sae_loss(X, f, X_hat, l1)`: mean over samples of ‖X − X̂‖² + l1 · ‖f‖₁.
- `topk_encode(pre, k)`: a top-k SAE's activation — keep the k largest pre-activations per row (after ReLU), zero
  the rest.
- `sae_metrics(X, f, X_hat)`: dict with `l0` (mean number of active latents per sample), `fve` (fraction of
  variance explained, 1 − Σ‖X − X̂‖² / Σ‖X − mean(X)‖²) and `dead` (fraction of latents never active).""",
starter='''import numpy as np

def normalize_decoder(W_dec):
    raise NotImplementedError

def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    raise NotImplementedError

def sae_loss(X, f, X_hat, l1):
    raise NotImplementedError

def topk_encode(pre, k):
    raise NotImplementedError

def sae_metrics(X, f, X_hat):
    raise NotImplementedError
''',
solution='''import numpy as np

def normalize_decoder(W_dec):
    W = np.asarray(W_dec, float)
    return W / np.linalg.norm(W, axis=1, keepdims=True)

def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    f = np.maximum((X - b_dec) @ W_enc + b_enc, 0.0)
    return f, f @ W_dec + b_dec

def sae_loss(X, f, X_hat, l1):
    return float(np.mean(np.sum((X - X_hat) ** 2, 1) + l1 * np.sum(np.abs(f), 1)))

def topk_encode(pre, k):
    pre = np.maximum(np.asarray(pre, float), 0.0)
    out = np.zeros_like(pre)
    idx = np.argsort(-pre, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(pre, idx, 1), 1)
    return out

def sae_metrics(X, f, X_hat):
    l0 = float(np.mean(np.sum(f > 0, 1)))
    fve = float(1 - np.sum((X - X_hat) ** 2) / np.sum((X - X.mean(0)) ** 2))
    dead = float(np.mean(~np.any(f > 0, 0)))
    return {"l0": l0, "fve": fve, "dead": dead}
''',
tests='''
def test_decoder_norm():
    W = normalize_decoder(np.random.default_rng(0).normal(size=(6, 4)))
    assert np.allclose(np.linalg.norm(W, axis=1), 1)

def test_perfect_dictionary_recovers_sparse_codes():
    rng = np.random.default_rng(1)
    d, m, n = 8, 8, 500
    D = np.eye(d)[rng.permutation(d)]                      # exact orthonormal features (a permutation)
    codes = (rng.random((n, m)) < 0.2) * rng.uniform(0.5, 2, (n, m))
    X = codes @ D
    f, Xh = sae_forward(X, D.T, np.zeros(m), D, np.zeros(d))
    assert np.allclose(f, codes) and np.allclose(Xh, X)
    m_ = sae_metrics(X, f, Xh)
    assert np.isclose(m_["fve"], 1) and np.isclose(m_["l0"], np.mean(np.sum(codes > 0, 1)))
    assert np.isclose(sae_loss(X, f, Xh, 0.1), 0.1 * np.mean(codes.sum(1)))

def test_topk():
    pre = np.array([[0.5, -1.0, 2.0, 1.0], [3.0, 2.0, 1.0, 0.0]])
    assert np.allclose(topk_encode(pre, 2), [[0, 0, 2, 1], [3, 2, 0, 0]])

def test_dead_latents():
    X = np.ones((4, 2)); f = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1], [1, 0, 0.0]])
    assert np.isclose(sae_metrics(X, f, X)["dead"], 1 / 3)
'''))
