"""Coding tasks, part 2 (Weeks 7–15)."""

T = []

T.append(dict(
id="w07_softmax_ce", title="Stable softmax cross-entropy and its gradient",
prompt="""Implement `softmax_cross_entropy(logits, y)` for logits of shape (N, C) and integer labels `y` (N,).

Return `(loss, dlogits)`: the **mean** cross-entropy over the batch and its gradient with respect to the logits.
It must stay finite for logits as large as ±1e4 (use the log-sum-exp trick).""",
starter='''import numpy as np

def softmax_cross_entropy(logits, y):
    raise NotImplementedError
''',
solution='''import numpy as np

def softmax_cross_entropy(logits, y):
    z = np.asarray(logits, float)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    p = np.exp(logp)
    p[np.arange(n), y] -= 1
    return float(loss), p / n
''',
tests='''
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
'''))

T.append(dict(
id="w07_mlp_backward", title="Two-layer MLP: forward and backward",
prompt="""Implement `mlp_loss_grads(X, y, W1, b1, W2, b2)` for the network
`h = relu(X @ W1 + b1)`, `logits = h @ W2 + b2`, with mean softmax cross-entropy loss.

Return `(loss, grads)` where `grads` is a dict with keys `"W1", "b1", "W2", "b2"` matching the parameter shapes.
Derive the backward pass yourself (no autodiff library).""",
starter='''import numpy as np

def mlp_loss_grads(X, y, W1, b1, W2, b2):
    raise NotImplementedError
''',
solution='''import numpy as np

def mlp_loss_grads(X, y, W1, b1, W2, b2):
    a = X @ W1 + b1
    h = np.maximum(a, 0)
    z = h @ W2 + b2
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    n = len(y)
    loss = -logp[np.arange(n), y].mean()
    dz = np.exp(logp); dz[np.arange(n), y] -= 1; dz /= n
    dW2 = h.T @ dz; db2 = dz.sum(0)
    dh = dz @ W2.T
    da = dh * (a > 0)
    dW1 = X.T @ da; db1 = da.sum(0)
    return float(loss), {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
''',
tests='''
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
'''))

T.append(dict(
id="w08_ext_gcd", title="Extended Euclid, modular inverse and CRT",
prompt="""Implement without `pow(a, -1, m)` or any library number theory:

- `ext_gcd(a, b)` → `(g, x, y)` with `g = gcd(a, b) ≥ 0` and `a*x + b*y = g`.
- `mod_inverse(a, m)` → the inverse of `a` modulo `m` in `[0, m)`; raise `ValueError` if it doesn't exist.
- `crt(residues, moduli)` → the smallest non-negative `x` with `x ≡ r_i (mod m_i)` for pairwise-coprime moduli.""",
starter='''def ext_gcd(a: int, b: int):
    raise NotImplementedError

def mod_inverse(a: int, m: int) -> int:
    raise NotImplementedError

def crt(residues, moduli) -> int:
    raise NotImplementedError
''',
solution='''def ext_gcd(a, b):
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    if old_r < 0:
        old_r, old_x, old_y = -old_r, -old_x, -old_y
    return old_r, old_x, old_y

def mod_inverse(a, m):
    g, x, _ = ext_gcd(a % m, m)
    if g != 1:
        raise ValueError("no inverse")
    return x % m

def crt(residues, moduli):
    x, M = 0, 1
    for m in moduli:
        M *= m
    for r, m in zip(residues, moduli):
        Mi = M // m
        x += r * Mi * mod_inverse(Mi, m)
    return x % M
''',
tests='''
import math, random
import pytest

def test_bezout_random():
    rnd = random.Random(0)
    for _ in range(500):
        a, b = rnd.randint(-10**6, 10**6), rnd.randint(-10**6, 10**6)
        g, x, y = ext_gcd(a, b)
        assert g == math.gcd(a, b) and a * x + b * y == g

def test_inverse():
    for m in [7, 26, 97, 1000]:
        for a in range(1, m):
            if math.gcd(a, m) == 1:
                assert (a * mod_inverse(a, m)) % m == 1
    with pytest.raises(ValueError):
        mod_inverse(6, 9)

def test_crt():
    assert crt([2, 3, 2], [3, 5, 7]) == 23
    rnd = random.Random(1)
    for _ in range(100):
        x = rnd.randint(0, 3 * 5 * 7 * 11 - 1)
        assert crt([x % 3, x % 5, x % 7, x % 11], [3, 5, 7, 11]) == x
'''))

T.append(dict(
id="w08_quantize_int8", title="Symmetric int8 quantization",
prompt="""Implement per-tensor and per-channel **symmetric** int8 quantization.

- `quantize(x, axis=None)` → `(q, scale)`: `q` is `np.int8` in [-127, 127], `scale = max|x| / 127` (per tensor,
  or per slice along `axis`, kept broadcastable). Use round-half-to-even (`np.round`). If max|x| is 0, use scale 1.
- `dequantize(q, scale)` → float array `q * scale`.
The round-trip error per element must be at most `scale / 2`.""",
starter='''import numpy as np

def quantize(x, axis=None):
    raise NotImplementedError

def dequantize(q, scale):
    raise NotImplementedError
''',
solution='''import numpy as np

def quantize(x, axis=None):
    x = np.asarray(x, float)
    if axis is None:
        amax = np.max(np.abs(x))
    else:
        amax = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 127.0)
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int8)
    return q, scale

def dequantize(q, scale):
    return q.astype(float) * scale
''',
tests='''
def test_dtype_and_range():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 8)) * 3
    q, s = quantize(x)
    assert q.dtype == np.int8 and q.min() >= -127 and q.max() <= 127
    assert np.isclose(np.max(np.abs(q)), 127)

def test_error_bound_per_tensor():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    q, s = quantize(x)
    assert np.all(np.abs(dequantize(q, s) - x) <= s / 2 + 1e-12)

def test_per_channel_better():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 64)) * np.array([[0.01], [0.1], [1.0], [10.0]])
    q1, s1 = quantize(x); q2, s2 = quantize(x, axis=1)
    assert np.asarray(s2).shape == (4, 1)
    e1 = np.abs(dequantize(q1, s1) - x)[0].mean(); e2 = np.abs(dequantize(q2, s2) - x)[0].mean()
    assert e2 < e1

def test_zero_tensor():
    q, s = quantize(np.zeros(5))
    assert np.all(q == 0) and np.all(dequantize(q, s) == 0)
'''))

T.append(dict(
id="w09_batchnorm", title="BatchNorm forward and backward (training mode)",
prompt="""Implement `batchnorm_forward_backward(x, gamma, beta, dout, eps=1e-5)` for `x` of shape (N, D).

Forward (training mode): normalize each feature with the **batch** mean and biased variance, then scale and shift.
Return `(out, dx, dgamma, dbeta)` where `dout` is the upstream gradient. Derive `dx` yourself.""",
starter='''import numpy as np

def batchnorm_forward_backward(x, gamma, beta, dout, eps=1e-5):
    raise NotImplementedError
''',
solution='''import numpy as np

def batchnorm_forward_backward(x, gamma, beta, dout, eps=1e-5):
    mu = x.mean(0); var = x.var(0)
    inv = 1 / np.sqrt(var + eps)
    xh = (x - mu) * inv
    out = gamma * xh + beta
    n = x.shape[0]
    dgamma = (dout * xh).sum(0); dbeta = dout.sum(0)
    dxh = dout * gamma
    dx = inv / n * (n * dxh - dxh.sum(0) - xh * (dxh * xh).sum(0))
    return out, dx, dgamma, dbeta
''',
tests='''
def test_forward_stats():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(64, 5))
    out, *_ = batchnorm_forward_backward(x, np.ones(5), np.zeros(5), np.zeros((64, 5)))
    assert np.allclose(out.mean(0), 0, atol=1e-10) and np.allclose(out.std(0), 1, atol=1e-3)

def test_backward_numeric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 3)); g = rng.normal(size=3); b = rng.normal(size=3); dout = rng.normal(size=(6, 3))
    _, dx, dg, db = batchnorm_forward_backward(x, g, b, dout)
    f = lambda x_, g_, b_: (batchnorm_forward_backward(x_, g_, b_, dout)[0] * dout).sum()
    num = np.zeros_like(x)
    for i in range(6):
        for j in range(3):
            e = np.zeros_like(x); e[i, j] = 1e-6
            num[i, j] = (f(x + e, g, b) - f(x - e, g, b)) / 2e-6
    assert np.allclose(dx, num, atol=1e-6)
    numg = np.array([(f(x, g + e, b) - f(x, g - e, b)) / 2e-6 for e in np.eye(3) * 1e-6])
    assert np.allclose(dg, numg, atol=1e-6) and np.allclose(db, dout.sum(0))
'''))

T.append(dict(
id="w09_warmup_cosine", title="Linear warmup + cosine decay schedule",
prompt="""Implement `lr_at(step, total_steps, warmup_steps, base_lr, min_lr=0.0)` (steps start at 0).

- Warmup: for `step < warmup_steps`, `lr = base_lr * (step + 1) / warmup_steps`, so it reaches `base_lr`
  at `step = warmup_steps - 1`.
- Then cosine decay from `base_lr` at `step = warmup_steps` to `min_lr` at `step = total_steps - 1`.
- For `step ≥ total_steps`, return `min_lr`.""",
starter='''import math

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    raise NotImplementedError
''',
solution='''import math

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr=0.0):
    if step >= total_steps:
        return min_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    span = max(1, total_steps - 1 - warmup_steps)
    p = (step - warmup_steps) / span
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))
''',
tests='''
import math

def test_warmup():
    assert abs(lr_at(0, 100, 10, 1.0) - 0.1) < 1e-12
    assert abs(lr_at(9, 100, 10, 1.0) - 1.0) < 1e-12

def test_peak_and_end():
    assert abs(lr_at(10, 100, 10, 1.0, 0.1) - 1.0) < 1e-12
    assert abs(lr_at(99, 100, 10, 1.0, 0.1) - 0.1) < 1e-12
    assert lr_at(150, 100, 10, 1.0, 0.1) == 0.1

def test_midpoint_and_monotone():
    mid = 10 + (99 - 10) / 2
    assert abs(lr_at(int(mid), 100, 10, 2.0) - (1 + math.cos(math.pi * (int(mid) - 10) / 89))) < 1e-12
    vals = [lr_at(s, 100, 10, 1.0) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:]))
'''))

T.append(dict(
id="w10_conv2d", title="Conv2d forward with stride and padding (im2col)",
prompt="""Implement `conv2d(x, w, b, stride=1, padding=0)` (cross-correlation, like deep-learning libraries).

Shapes: `x` (N, C, H, W), `w` (F, C, KH, KW), `b` (F,). Output (N, F, H_out, W_out) with
`H_out = (H + 2·padding − KH) // stride + 1` (same for width). Use an im2col-style vectorized approach
rather than six nested loops.""",
starter='''import numpy as np

def conv2d(x, w, b, stride=1, padding=0):
    raise NotImplementedError
''',
solution='''import numpy as np

def conv2d(x, w, b, stride=1, padding=0):
    N, C, H, W = x.shape
    F, _, KH, KW = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho = (H + 2 * padding - KH) // stride + 1
    Wo = (W + 2 * padding - KW) // stride + 1
    cols = np.empty((N, C, KH, KW, Ho, Wo))
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j] = xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
    out = np.einsum("nckluv,fckl->nfuv", cols, w) + b[None, :, None, None]
    return out
''',
tests='''
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
'''))

T.append(dict(
id="w10_fft", title="Recursive radix-2 FFT",
prompt="""Implement `fft(x)` for a 1-D array whose length is a power of two, using the Cooley–Tukey recursion
(split into even and odd samples). Return a complex array equal to `np.fft.fft(x)`. Also implement `ifft(X)`
using your `fft` (hint: conjugation). Don't call `np.fft`.""",
starter='''import numpy as np

def fft(x):
    raise NotImplementedError

def ifft(X):
    raise NotImplementedError
''',
solution='''import numpy as np

def fft(x):
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if n == 1:
        return x.copy()
    even, odd = fft(x[0::2]), fft(x[1::2])
    tw = np.exp(-2j * np.pi * np.arange(n // 2) / n) * odd
    return np.concatenate([even + tw, even - tw])

def ifft(X):
    X = np.asarray(X, dtype=complex)
    return np.conj(fft(np.conj(X))) / len(X)
''',
tests='''
def test_matches_numpy():
    rng = np.random.default_rng(0)
    for n in [1, 2, 8, 64, 256]:
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        assert np.allclose(fft(x), np.fft.fft(x))

def test_roundtrip():
    rng = np.random.default_rng(1)
    x = rng.normal(size=128)
    assert np.allclose(ifft(fft(x)), x)

def test_convolution_theorem():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=32), rng.normal(size=32)
    circ = np.array([sum(a[k] * b[(n - k) % 32] for k in range(32)) for n in range(32)])
    assert np.allclose(ifft(fft(a) * fft(b)).real, circ)
'''))

T.append(dict(
id="w11_stft", title="Short-time Fourier transform magnitude",
prompt="""Implement `stft_mag(x, n_fft, hop)`:

- Split `x` into frames of length `n_fft` starting at 0, hop, 2·hop, … (no padding; drop the incomplete tail).
- Multiply each frame by a periodic Hann window `w[n] = 0.5 − 0.5·cos(2πn / n_fft)`.
- Return the magnitude of the one-sided spectrum, shape `(n_fft // 2 + 1, n_frames)`.
You may use `np.fft.rfft` here.""",
starter='''import numpy as np

def stft_mag(x, n_fft: int, hop: int):
    raise NotImplementedError
''',
solution='''import numpy as np

def stft_mag(x, n_fft, hop):
    x = np.asarray(x, float)
    n_frames = 1 + (len(x) - n_fft) // hop
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * w
    return np.abs(np.fft.rfft(frames, axis=1)).T
''',
tests='''
def test_shape():
    S = stft_mag(np.zeros(1000), 256, 128)
    assert S.shape == (129, 1 + (1000 - 256) // 128)

def test_pure_tone_peak():
    sr, f = 8000, 1000
    t = np.arange(sr) / sr
    S = stft_mag(np.sin(2 * np.pi * f * t), 512, 256)
    peak_bin = np.argmax(S.mean(1))
    assert abs(peak_bin * sr / 512 - f) <= sr / 512

def test_first_frame_value():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(64) / 64)
    assert np.allclose(stft_mag(x, 64, 32)[:, 0], np.abs(np.fft.rfft(x[:64] * w)))
'''))

T.append(dict(
id="w11_kalman_1d", title="1-D Kalman filter (random-walk model)",
prompt="""Implement `kalman_1d(zs, q, r, x0, p0)` for the model `x_t = x_{t−1} + w`, `w ~ N(0, q)`, and
measurement `z_t = x_t + v`, `v ~ N(0, r)`.

For each measurement: predict (`p ← p + q`), then update with gain `k = p / (p + r)`.
Return `(xs, ps)`: arrays of the posterior mean and variance after each measurement.""",
starter='''import numpy as np

def kalman_1d(zs, q: float, r: float, x0: float, p0: float):
    raise NotImplementedError
''',
solution='''import numpy as np

def kalman_1d(zs, q, r, x0, p0):
    x, p = float(x0), float(p0)
    xs, ps = [], []
    for z in zs:
        p = p + q
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        xs.append(x); ps.append(p)
    return np.array(xs), np.array(ps)
''',
tests='''
def test_static_signal_equals_running_mean():
    # q = 0: the filter is a precision-weighted running average of prior and measurements.
    rng = np.random.default_rng(0)
    z = 5 + rng.normal(size=50)
    p0, x0 = 1e3, 0.0
    n = np.arange(1, 51)
    xs, ps = kalman_1d(z, 0.0, 1.0, x0, p0)
    assert np.allclose(xs, (x0 / p0 + np.cumsum(z)) / (1 / p0 + n), atol=1e-9)
    assert np.allclose(ps, 1 / (1 / p0 + n), atol=1e-9)

def test_trusts_precise_sensor():
    xs, _ = kalman_1d([3.0, -2.0], 1.0, 1e-12, 0.0, 1.0)
    assert np.allclose(xs, [3.0, -2.0], atol=1e-6)

def test_variance_converges():
    _, ps = kalman_1d(np.zeros(200), 0.1, 1.0, 0.0, 10.0)
    q, r = 0.1, 1.0
    p_pred = (q + np.sqrt(q * q + 4 * q * r)) / 2   # steady-state predicted variance
    assert abs(ps[-1] - p_pred * r / (p_pred + r)) < 1e-6
'''))

T.append(dict(
id="w12_label_propagation", title="Label propagation: the harmonic solution",
prompt="""Implement `label_propagation(W, labeled_idx, Y_l)`.

- `W`: symmetric (n, n) non-negative affinity matrix; `labeled_idx`: indices of labelled nodes; `Y_l`: one-hot
  labels (len(labeled_idx), C).
- Solve the harmonic problem: unlabeled scores `F_u = (D_uu − W_uu)^{-1} W_ul Y_l` with `D` the degree matrix.
- Return `F` of shape (n, C): labelled rows equal `Y_l`, unlabeled rows `F_u`. Rows sum to 1 when every
  unlabeled node is connected to a labelled one.""",
starter='''import numpy as np

def label_propagation(W, labeled_idx, Y_l):
    raise NotImplementedError
''',
solution='''import numpy as np

def label_propagation(W, labeled_idx, Y_l):
    W = np.asarray(W, float); n = len(W)
    L = np.asarray(labeled_idx)
    U = np.setdiff1d(np.arange(n), L)
    D = np.diag(W.sum(1))
    Luu = (D - W)[np.ix_(U, U)]
    Fu = np.linalg.solve(Luu, W[np.ix_(U, L)] @ Y_l)
    F = np.zeros((n, Y_l.shape[1]))
    F[L] = Y_l; F[U] = Fu
    return F
''',
tests='''
def test_chain_interpolates():
    n = 5
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1
    F = label_propagation(W, [0, 4], np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(F[:, 0], [1, 0.75, 0.5, 0.25, 0])
    assert np.allclose(F.sum(1), 1)

def test_two_clusters():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 0.3, (10, 2)); B = rng.normal(5, 0.3, (10, 2))
    X = np.vstack([A, B])
    d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    W = np.exp(-d2); np.fill_diagonal(W, 0)
    F = label_propagation(W, [0, 10], np.eye(2))
    assert np.all(F[:10].argmax(1) == 0) and np.all(F[10:].argmax(1) == 1)

def test_harmonic_property():
    rng = np.random.default_rng(1)
    W = rng.random((7, 7)); W = W + W.T; np.fill_diagonal(W, 0)
    F = label_propagation(W, [0, 1], np.eye(2))
    for u in range(2, 7):
        assert np.allclose(F[u], W[u] @ F / W[u].sum())
'''))

T.append(dict(
id="w12_fixmatch_targets", title="FixMatch pseudo-labels and unlabeled loss",
prompt="""Implement the unlabeled part of FixMatch.

`fixmatch_unlabeled_loss(probs_weak, logits_strong, threshold=0.95)`:
- `probs_weak` (N, C): predicted probabilities on weakly augmented images (treat as constants).
- Pseudo-label = argmax of `probs_weak`; mask = max probability ≥ threshold.
- Loss = cross-entropy of `logits_strong` against the pseudo-labels, **summed over masked examples and divided
  by N** (all unlabeled examples, not just the confident ones).
Return `(loss, mask, pseudo_labels)`. Use a numerically stable log-softmax.""",
starter='''import numpy as np

def fixmatch_unlabeled_loss(probs_weak, logits_strong, threshold=0.95):
    raise NotImplementedError
''',
solution='''import numpy as np

def fixmatch_unlabeled_loss(probs_weak, logits_strong, threshold=0.95):
    p = np.asarray(probs_weak, float)
    z = np.asarray(logits_strong, float)
    pl = p.argmax(1)
    mask = p.max(1) >= threshold
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    ce = -logp[np.arange(len(pl)), pl]
    return float((ce * mask).sum() / len(pl)), mask, pl
''',
tests='''
def test_mask_and_labels():
    pw = np.array([[0.97, 0.03], [0.6, 0.4], [0.01, 0.99]])
    loss, mask, pl = fixmatch_unlabeled_loss(pw, np.zeros((3, 2)), 0.95)
    assert list(mask) == [True, False, True] and list(pl) == [0, 0, 1]
    assert abs(loss - 2 * np.log(2) / 3) < 1e-12

def test_no_confident_examples():
    loss, mask, _ = fixmatch_unlabeled_loss(np.full((4, 3), 1 / 3), np.random.default_rng(0).normal(size=(4, 3)))
    assert loss == 0.0 and not mask.any()

def test_stable():
    pw = np.array([[1.0, 0.0]]); zs = np.array([[-1e4, 1e4]])
    loss, _, _ = fixmatch_unlabeled_loss(pw, zs)
    assert np.isfinite(loss) and abs(loss - 2e4) < 1e-6
'''))

T.append(dict(
id="w13_info_nce", title="Symmetric InfoNCE (NT-Xent style) loss",
prompt="""Implement `info_nce(z1, z2, temperature=0.1)` for two batches of embeddings (N, D) where row i of `z1` and
row i of `z2` are a positive pair.

- L2-normalize rows, compute similarities `S = z1 @ z2.T / temperature`.
- Loss = mean of the cross-entropy over rows (target = diagonal) and over columns, i.e.
  `0.5 * (CE(S, diag) + CE(S.T, diag))`.
Use a stable log-softmax.""",
starter='''import numpy as np

def info_nce(z1, z2, temperature=0.1) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def _ce_diag(S):
    S = S - S.max(1, keepdims=True)
    logp = S - np.log(np.exp(S).sum(1, keepdims=True))
    return -np.mean(np.diag(logp))

def info_nce(z1, z2, temperature=0.1):
    a = z1 / np.linalg.norm(z1, axis=1, keepdims=True)
    b = z2 / np.linalg.norm(z2, axis=1, keepdims=True)
    S = a @ b.T / temperature
    return float(0.5 * (_ce_diag(S) + _ce_diag(S.T)))
''',
tests='''
def test_random_is_about_log_n():
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=(64, 128)); z2 = rng.normal(size=(64, 128))
    assert abs(info_nce(z1, z2, temperature=100.0) - np.log(64)) < 0.01

def test_aligned_pairs_low_loss():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(16, 32))
    assert info_nce(z, z + 0.01 * rng.normal(size=z.shape), 0.05) < 0.05

def test_scale_invariant_and_symmetric():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(8, 4)), rng.normal(size=(8, 4))
    assert abs(info_nce(a, b) - info_nce(3 * a, 0.5 * b)) < 1e-10
    assert abs(info_nce(a, b) - info_nce(b, a)) < 1e-10
'''))

T.append(dict(
id="w13_effective_rank", title="Effective rank of an embedding matrix",
prompt="""Implement `effective_rank(Z)` (Roy & Vetterli): center `Z` (n, d) by column means, take its singular
values `s`, form `p = s / s.sum()`, and return `exp(H(p))` with `H(p) = −Σ p log p` (treat 0·log 0 as 0).
A collapsed representation has effective rank near 1.""",
starter='''import numpy as np

def effective_rank(Z) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def effective_rank(Z):
    Z = np.asarray(Z, float)
    Z = Z - Z.mean(0)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))
''',
tests='''
def test_isotropic_full_rank():
    rng = np.random.default_rng(0)
    Z = np.linalg.qr(rng.normal(size=(200, 200)))[0][:, :10]
    Z = Z - Z.mean(0)
    assert abs(effective_rank(Z) - 10) < 0.2

def test_rank_one():
    rng = np.random.default_rng(1)
    Z = np.outer(rng.normal(size=100), rng.normal(size=8))
    assert abs(effective_rank(Z) - 1) < 1e-6

def test_between():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 6)) * np.array([10, 1, 1, 0.1, 0.1, 0.1])
    r = effective_rank(Z)
    assert 1 < r < 6
'''))

T.append(dict(
id="w14_bpe_merges", title="Byte-pair encoding: learn merges",
prompt="""Implement `learn_bpe(word_counts, num_merges)`.

- `word_counts`: dict word → count. Each word starts as a tuple of its characters plus the end marker `"</w>"`.
- Repeat `num_merges` times: count adjacent symbol pairs weighted by word counts; merge the most frequent pair
  everywhere. **Break ties by the lexicographically smallest pair** (compare the tuple `(a, b)`). Stop early if no
  pair remains.
- Return the list of merges as tuples `(a, b)` in order.""",
starter='''def learn_bpe(word_counts: dict, num_merges: int) -> list:
    raise NotImplementedError
''',
solution='''from collections import Counter

def learn_bpe(word_counts, num_merges):
    vocab = {tuple(w) + ("</w>",): c for w, c in word_counts.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = min(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        merges.append(best)
        new = {}
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        vocab = new
    return merges
''',
tests='''
def test_classic_example():
    counts = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    m = learn_bpe(counts, 4)
    assert m[0] == ("e", "s")
    assert m[1] == ("es", "t")
    assert m[2] == ("est", "</w>")
    assert m[3] == ("l", "o")

def test_tie_break():
    assert learn_bpe({"ab": 1, "cd": 1}, 1) == [("a", "b")]

def test_stops_early():
    assert len(learn_bpe({"a": 3}, 10)) == 1
'''))

T.append(dict(
id="w14_lstm_cell", title="LSTM cell forward",
prompt="""Implement one LSTM step `lstm_cell(x, h, c, W, U, b)` with batch inputs.

Shapes: `x` (N, D), `h` and `c` (N, H), `W` (D, 4H), `U` (H, 4H), `b` (4H,). Pre-activations
`a = x W + h U + b` are split in the order **input, forget, cell (g), output** along the last axis.
`i = σ(a_i)`, `f = σ(a_f)`, `g = tanh(a_g)`, `o = σ(a_o)`, `c' = f⊙c + i⊙g`, `h' = o⊙tanh(c')`.
Return `(h_new, c_new)`.""",
starter='''import numpy as np

def lstm_cell(x, h, c, W, U, b):
    raise NotImplementedError
''',
solution='''import numpy as np

def _sig(z):
    return 1 / (1 + np.exp(-z))

def lstm_cell(x, h, c, W, U, b):
    a = x @ W + h @ U + b
    H = h.shape[1]
    i, f, g, o = a[:, :H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
    c2 = _sig(f) * c + _sig(i) * np.tanh(g)
    return _sig(o) * np.tanh(c2), c2
''',
tests='''
def test_shapes_and_reference():
    rng = np.random.default_rng(0)
    N, D, H = 3, 4, 5
    x, h, c = rng.normal(size=(N, D)), rng.normal(size=(N, H)), rng.normal(size=(N, H))
    W, U, b = rng.normal(size=(D, 4*H)), rng.normal(size=(H, 4*H)), rng.normal(size=4*H)
    h2, c2 = lstm_cell(x, h, c, W, U, b)
    s = lambda z: 1 / (1 + np.exp(-z))
    a = x @ W + h @ U + b
    ci = s(a[:, H:2*H]) * c + s(a[:, :H]) * np.tanh(a[:, 2*H:3*H])
    assert np.allclose(c2, ci) and np.allclose(h2, s(a[:, 3*H:]) * np.tanh(ci))

def test_forget_gate_keeps_memory():
    H = 2
    W = np.zeros((1, 4*H)); U = np.zeros((H, 4*H))
    b = np.concatenate([np.full(H, -50.0), np.full(H, 50.0), np.zeros(H), np.full(H, 50.0)])
    c = np.array([[0.3, -0.7]])
    h2, c2 = lstm_cell(np.zeros((1, 1)), np.zeros((1, H)), c, W, U, b)
    assert np.allclose(c2, c) and np.allclose(h2, np.tanh(c))
'''))

T.append(dict(
id="w15_attention", title="Scaled dot-product attention with masking",
prompt="""Implement `attention(Q, K, V, mask=None)` for batched inputs `Q` (..., Lq, d), `K` (..., Lk, d),
`V` (..., Lk, dv). Scores are `Q Kᵀ / √d`. `mask` is boolean, broadcastable to (..., Lq, Lk), with **True meaning
"may attend"**; masked-out scores get −∞ before the softmax. Return `(out, weights)`.
Also implement `causal_mask(L)` returning an (L, L) boolean lower-triangular mask.""",
starter='''import numpy as np

def causal_mask(L: int):
    raise NotImplementedError

def attention(Q, K, V, mask=None):
    raise NotImplementedError
''',
solution='''import numpy as np

def causal_mask(L):
    return np.tril(np.ones((L, L), dtype=bool))

def attention(Q, K, V, mask=None):
    d = Q.shape[-1]
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    w = np.exp(s)
    w = w / w.sum(-1, keepdims=True)
    return w @ V, w
''',
tests='''
def test_weights_sum_to_one_and_shape():
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 7, 8)), rng.normal(size=(2, 3, 7, 4))
    out, w = attention(Q, K, V)
    assert out.shape == (2, 3, 5, 4) and np.allclose(w.sum(-1), 1)

def test_causal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 4))
    out, w = attention(X, X, X, causal_mask(6))
    assert np.allclose(np.triu(w, 1), 0)
    assert np.allclose(out[0], X[0])

def test_scaling():
    q = np.array([[1.0, 1.0, 1.0, 1.0]]); k = np.array([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    _, w = attention(q, k, np.eye(2))
    e = np.exp(4 / 2)
    assert np.allclose(w[0], [e / (e + 1), 1 / (e + 1)])
'''))

T.append(dict(
id="w15_rope", title="Rotary position embedding (RoPE)",
prompt="""Implement `apply_rope(x, positions, base=10000.0)` for `x` of shape (L, d) with even d.

Pair dimensions as (0,1), (2,3), …; pair k uses frequency `θ_k = base^(−2k/d)` and is rotated by angle
`positions[t] · θ_k`: `(x0, x1) → (x0 cos − x1 sin, x0 sin + x1 cos)`. Return the rotated array.
A correct implementation makes `apply_rope(q, [m]) · apply_rope(k, [n])` depend only on `m − n`.""",
starter='''import numpy as np

def apply_rope(x, positions, base=10000.0):
    raise NotImplementedError
''',
solution='''import numpy as np

def apply_rope(x, positions, base=10000.0):
    x = np.asarray(x, float)
    L, d = x.shape
    k = np.arange(d // 2)
    theta = base ** (-2 * k / d)
    ang = np.asarray(positions, float)[:, None] * theta[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x0, x1 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x0 * c - x1 * s
    out[:, 1::2] = x0 * s + x1 * c
    return out
''',
tests='''
def test_norm_preserved():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 16))
    y = apply_rope(x, np.arange(10))
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))

def test_position_zero_identity():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 8))
    assert np.allclose(apply_rope(x, [0, 0, 0]), x)

def test_relative_property():
    rng = np.random.default_rng(2)
    q, k = rng.normal(size=(1, 32)), rng.normal(size=(1, 32))
    s1 = apply_rope(q, [7]) @ apply_rope(k, [3]).T
    s2 = apply_rope(q, [104]) @ apply_rope(k, [100]).T
    assert np.allclose(s1, s2)

def test_first_pair_angle():
    x = np.array([[1.0, 0.0, 0.0, 0.0]])
    y = apply_rope(x, [2.0])
    assert np.allclose(y[0, :2], [np.cos(2.0), np.sin(2.0)])
'''))
