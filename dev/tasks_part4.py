"""Coding tasks, part 4 (Weeks 9, 10B, 11B, 13B)."""

T = []

T.append(dict(
id="w10b_gaussian_sobel", title="Separable Gaussian blur and Sobel gradients",
prompt="""Implement classical filters for grayscale images (2-D float arrays).

- `gaussian_kernel1d(sigma)`: radius `r = ceil(3·sigma)`, weights `exp(−x²/2σ²)` for x = −r…r, normalized to sum 1.
- `gaussian_blur(img, sigma)`: separable blur (rows then columns) with **reflect** padding
  (`np.pad(..., mode="reflect")`, i.e. the edge pixel is not repeated). Output has the input's shape.
- `sobel(img)`: return `(gx, gy, magnitude)` using the 3×3 Sobel kernels as **correlation**
  (`gx` responds to left→right increases), with `np.pad(..., mode="symmetric")` padding.""",
starter='''import numpy as np

def gaussian_kernel1d(sigma: float) -> np.ndarray:
    raise NotImplementedError

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    raise NotImplementedError

def sobel(img: np.ndarray):
    raise NotImplementedError
''',
solution='''import numpy as np

def gaussian_kernel1d(sigma):
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()

def _corr1d(img, k, axis, mode):
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]; pad[axis] = (r, r)
    p = np.pad(img, pad, mode=mode)
    out = np.zeros_like(img, dtype=float)
    n = img.shape[axis]
    for i, w in enumerate(k):
        sl = [slice(None), slice(None)]; sl[axis] = slice(i, i + n)
        out += w * p[tuple(sl)]
    return out

def gaussian_blur(img, sigma):
    k = gaussian_kernel1d(sigma)
    return _corr1d(_corr1d(np.asarray(img, float), k, 0, "reflect"), k, 1, "reflect")

def sobel(img):
    img = np.asarray(img, float)
    d = np.array([-1.0, 0.0, 1.0]); s = np.array([1.0, 2.0, 1.0])
    gx = _corr1d(_corr1d(img, d, 1, "symmetric"), s, 0, "symmetric")
    gy = _corr1d(_corr1d(img, d, 0, "symmetric"), s, 1, "symmetric")
    return gx, gy, np.hypot(gx, gy)
''',
tests='''
from scipy import ndimage

def test_kernel():
    k = gaussian_kernel1d(1.5)
    assert len(k) == 2 * 5 + 1 and np.isclose(k.sum(), 1) and np.allclose(k, k[::-1])

def test_blur_equals_2d_correlation():
    rng = np.random.default_rng(0)
    img = rng.random((20, 17))
    k = gaussian_kernel1d(1.2)
    ref = ndimage.correlate(img, np.outer(k, k), mode="mirror")
    assert np.allclose(gaussian_blur(img, 1.2), ref)

def test_blur_keeps_mean_of_constant():
    assert np.allclose(gaussian_blur(np.full((8, 8), 3.0), 2.0), 3.0)

def test_sobel_matches_scipy():
    rng = np.random.default_rng(1)
    img = rng.random((15, 12))
    gx, gy, mag = sobel(img)
    assert np.allclose(gx, ndimage.sobel(img, axis=1, mode="reflect"))
    assert np.allclose(gy, ndimage.sobel(img, axis=0, mode="reflect"))
    assert np.allclose(mag, np.hypot(gx, gy))

def test_step_edge_direction():
    img = np.zeros((9, 9)); img[:, 5:] = 1
    gx, gy, _ = sobel(img)
    assert gx[4, 4] > 0 and np.allclose(gy, 0)
'''))

T.append(dict(
id="w10b_conv_variants", title="Convolution arithmetic: dilated and transposed convolution",
prompt="""Single-channel 2-D operations (valid cross-correlation, no padding):

- `conv_out_size(n, k, stride=1, padding=0, dilation=1)`: the standard output-length formula.
- `dilated_conv2d(x, w, dilation)`: valid correlation of `x` with kernel `w` whose taps are spaced `dilation` apart.
- `strided_conv2d(x, w, stride)`: valid correlation sampled every `stride` pixels.
- `conv_transpose2d(y, w, stride, out_shape)`: the **adjoint** of `strided_conv2d` for an input of shape
  `out_shape`, so that `⟨strided_conv2d(x, w, s), y⟩ = ⟨x, conv_transpose2d(y, w, s, x.shape)⟩` for all x, y.""",
starter='''import numpy as np

def conv_out_size(n, k, stride=1, padding=0, dilation=1):
    raise NotImplementedError

def dilated_conv2d(x, w, dilation):
    raise NotImplementedError

def strided_conv2d(x, w, stride):
    raise NotImplementedError

def conv_transpose2d(y, w, stride, out_shape):
    raise NotImplementedError
''',
solution='''import numpy as np

def conv_out_size(n, k, stride=1, padding=0, dilation=1):
    return (n + 2 * padding - dilation * (k - 1) - 1) // stride + 1

def dilated_conv2d(x, w, dilation):
    kh, kw = w.shape
    H = conv_out_size(x.shape[0], kh, 1, 0, dilation); W = conv_out_size(x.shape[1], kw, 1, 0, dilation)
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += w[i, j] * x[i * dilation:i * dilation + H, j * dilation:j * dilation + W]
    return out

def strided_conv2d(x, w, stride):
    kh, kw = w.shape
    H = conv_out_size(x.shape[0], kh, stride); W = conv_out_size(x.shape[1], kw, stride)
    out = np.zeros((H, W))
    for i in range(kh):
        for j in range(kw):
            out += w[i, j] * x[i:i + stride * (H - 1) + 1:stride, j:j + stride * (W - 1) + 1:stride]
    return out

def conv_transpose2d(y, w, stride, out_shape):
    kh, kw = w.shape
    H, W = y.shape
    x = np.zeros(out_shape)
    for i in range(kh):
        for j in range(kw):
            x[i:i + stride * (H - 1) + 1:stride, j:j + stride * (W - 1) + 1:stride] += w[i, j] * y
    return x
''',
tests='''
from scipy import ndimage

def test_output_sizes():
    assert conv_out_size(32, 3) == 30
    assert conv_out_size(32, 3, stride=2, padding=1) == 16
    assert conv_out_size(32, 3, dilation=2) == 28
    assert conv_out_size(7, 3, stride=2) == 3

def test_dilated_equals_sparse_kernel():
    rng = np.random.default_rng(0)
    x, w = rng.random((12, 11)), rng.random((3, 3))
    big = np.zeros((5, 5)); big[::2, ::2] = w
    assert np.allclose(dilated_conv2d(x, w, 2), strided_conv2d(x, big, 1))

def test_strided_matches_scipy():
    rng = np.random.default_rng(1)
    x, w = rng.random((10, 9)), rng.random((3, 3))
    full = ndimage.correlate(x, w, mode="constant")[1:-1, 1:-1]
    assert np.allclose(strided_conv2d(x, w, 2), full[::2, ::2])

def test_transpose_is_adjoint():
    rng = np.random.default_rng(2)
    for s in (1, 2, 3):
        x, w = rng.random((11, 10)), rng.random((3, 3))
        y = rng.random(strided_conv2d(x, w, s).shape)
        lhs = (strided_conv2d(x, w, s) * y).sum()
        rhs = (x * conv_transpose2d(y, w, s, x.shape)).sum()
        assert np.isclose(lhs, rhs)
'''))

T.append(dict(
id="w11b_yin_pitch", title="YIN pitch estimation",
prompt="""Implement `yin_pitch(frame, sr, fmin=60.0, fmax=800.0, threshold=0.1)` returning the fundamental frequency in
Hz, or `0.0` for silence/unvoiced frames.

1. Difference function `d(τ) = Σ_j (x[j] − x[j+τ])²` over a fixed window `j = 0 … W−1` with `W = len(frame) − τ_max`.
2. Cumulative mean normalized difference `d′(0) = 1`, `d′(τ) = d(τ) · τ / Σ_{k=1..τ} d(k)`.
3. Search `τ ∈ [sr/fmax, sr/fmin]`: take the first τ with `d′(τ) < threshold`, then walk to the local minimum.
   If none falls below the threshold, use the global minimum only if `d′ < 0.5`, else return 0.
4. Refine τ with parabolic interpolation on `d′`, return `sr / τ`. Return 0 if the frame energy is ~0.""",
starter='''import numpy as np

def yin_pitch(frame, sr, fmin=60.0, fmax=800.0, threshold=0.1) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def yin_pitch(frame, sr, fmin=60.0, fmax=800.0, threshold=0.1):
    x = np.asarray(frame, float)
    if np.sum(x * x) < 1e-10:
        return 0.0
    tmin, tmax = int(sr / fmax), int(np.ceil(sr / fmin))
    W = len(x) - tmax
    if W <= 0:
        raise ValueError("frame too short for fmin")
    d = np.array([np.sum((x[:W] - x[t:t + W]) ** 2) for t in range(tmax + 1)])
    cm = np.ones_like(d)
    cs = np.cumsum(d[1:])
    cm[1:] = d[1:] * np.arange(1, tmax + 1) / np.where(cs == 0, 1e-12, cs)
    tau = None
    for t in range(max(tmin, 1), tmax):
        if cm[t] < threshold:
            while t + 1 < tmax and cm[t + 1] < cm[t]:
                t += 1
            tau = t
            break
    if tau is None:
        t = max(tmin, 1) + int(np.argmin(cm[max(tmin, 1):tmax]))
        if cm[t] >= 0.5:
            return 0.0
        tau = t
    if 1 <= tau < tmax:
        a, b, c = cm[tau - 1], cm[tau], cm[tau + 1]
        den = a - 2 * b + c
        shift = 0.5 * (a - c) / den if den != 0 else 0.0
        tau = tau + float(np.clip(shift, -1, 1))
    return float(sr / tau)
''',
tests='''
SR = 16000

def _tone(f, n=2048, harmonics=(1.0,)):
    t = np.arange(n) / SR
    return sum(a * np.sin(2 * np.pi * f * (k + 1) * t) for k, a in enumerate(harmonics))

def test_pure_tones():
    for f in (110.0, 220.0, 440.0):
        assert abs(yin_pitch(_tone(f), SR) - f) / f < 0.01

def test_no_octave_error_with_harmonics():
    x = _tone(150.0, harmonics=(0.4, 1.0, 0.8, 0.5))   # strong 2nd harmonic
    assert abs(yin_pitch(x, SR) - 150.0) / 150.0 < 0.02

def test_silence():
    assert yin_pitch(np.zeros(2048), SR) == 0.0
'''))

T.append(dict(
id="w11b_wiener_sisdr", title="Wiener gain, SNR and SI-SDR",
prompt="""Implement three audio-enhancement building blocks.

- `wiener_gain(noisy_power, noise_power, floor=0.0)`: per-bin gain using the maximum-likelihood a-priori SNR
  `ξ = max(P_noisy / P_noise − 1, 0)` and the Wiener rule `ξ / (1 + ξ)`, then clipped below at `floor`.
- `snr_db(clean, estimate)`: `10·log10(‖clean‖² / ‖clean − estimate‖²)`.
- `si_sdr(reference, estimate)`: scale-invariant SDR in dB — project the (zero-mean) estimate onto the (zero-mean)
  reference, `s_target = (⟨ŝ, s⟩ / ‖s‖²)·s`, `e = ŝ − s_target`, return `10·log10(‖s_target‖² / ‖e‖²)`.""",
starter='''import numpy as np

def wiener_gain(noisy_power, noise_power, floor=0.0):
    raise NotImplementedError

def snr_db(clean, estimate) -> float:
    raise NotImplementedError

def si_sdr(reference, estimate) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def wiener_gain(noisy_power, noise_power, floor=0.0):
    P = np.asarray(noisy_power, float); N = np.asarray(noise_power, float)
    xi = np.maximum(P / N - 1.0, 0.0)
    return np.maximum(xi / (1.0 + xi), floor)

def snr_db(clean, estimate):
    c = np.asarray(clean, float); e = c - np.asarray(estimate, float)
    return float(10 * np.log10(np.sum(c ** 2) / np.sum(e ** 2)))

def si_sdr(reference, estimate):
    s = np.asarray(reference, float); s = s - s.mean()
    h = np.asarray(estimate, float); h = h - h.mean()
    target = (h @ s) / (s @ s) * s
    e = h - target
    return float(10 * np.log10((target @ target) / (e @ e)))
''',
tests='''
def test_gain_properties():
    P = np.array([1.0, 2.0, 11.0, 0.5]); N = np.ones(4)
    g = wiener_gain(P, N)
    assert np.allclose(g, [0.0, 0.5, 10 / 11, 0.0])
    assert np.allclose(wiener_gain(P, N, floor=0.1), np.maximum(g, 0.1))

def test_gain_equals_power_subtraction():
    rng = np.random.default_rng(0)
    N = rng.random(50) + 0.1; P = N * (1 + 5 * rng.random(50))
    assert np.allclose(wiener_gain(P, N), 1 - N / P)

def test_snr_known():
    rng = np.random.default_rng(1)
    s = rng.normal(size=8000); n = rng.normal(size=8000)
    n = n / np.linalg.norm(n) * np.linalg.norm(s) / 10    # 20 dB
    assert np.isclose(snr_db(s, s + n), 20.0)

def test_si_sdr_scale_invariant():
    rng = np.random.default_rng(2)
    s = rng.normal(size=4000); n = rng.normal(size=4000) * 0.1
    a = si_sdr(s, s + n)
    assert np.isclose(a, si_sdr(s, 3.7 * (s + n)))
    assert si_sdr(s, s + 0.01 * n) > a
'''))

T.append(dict(
id="w13b_box_coding", title="Box encoding, IoU and GIoU",
prompt="""Boxes are `(x1, y1, x2, y2)` rows. Implement:

- `box_iou(a, b)`: (N, M) IoU matrix.
- `generalized_iou(a, b)`: element-wise GIoU for paired boxes of equal shape (N, 4):
  `IoU − |C \\ (A ∪ B)| / |C|` with C the smallest enclosing box.
- `encode(boxes, anchors)`: Faster R-CNN deltas `(tx, ty, tw, th)` with `tx = (x − xa)/wa`, `ty = (y − ya)/ha`,
  `tw = log(w / wa)`, `th = log(h / ha)` using box centers and widths.
- `decode(deltas, anchors)`: the inverse, returning xyxy boxes.""",
starter='''import numpy as np

def box_iou(a, b):
    raise NotImplementedError

def generalized_iou(a, b):
    raise NotImplementedError

def encode(boxes, anchors):
    raise NotImplementedError

def decode(deltas, anchors):
    raise NotImplementedError
''',
solution='''import numpy as np

def _area(b):
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

def box_iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2]); rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (_area(a)[:, None] + _area(b)[None, :] - inter)

def generalized_iou(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    lt = np.maximum(a[:, :2], b[:, :2]); rb = np.minimum(a[:, 2:], b[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    union = _area(a) + _area(b) - inter
    C = np.concatenate([np.minimum(a[:, :2], b[:, :2]), np.maximum(a[:, 2:], b[:, 2:])], 1)
    c = _area(C)
    return inter / union - (c - union) / c

def _cwh(b):
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    return b[:, 0] + w / 2, b[:, 1] + h / 2, w, h

def encode(boxes, anchors):
    x, y, w, h = _cwh(np.asarray(boxes, float)); xa, ya, wa, ha = _cwh(np.asarray(anchors, float))
    return np.stack([(x - xa) / wa, (y - ya) / ha, np.log(w / wa), np.log(h / ha)], 1)

def decode(deltas, anchors):
    d = np.asarray(deltas, float); xa, ya, wa, ha = _cwh(np.asarray(anchors, float))
    x = d[:, 0] * wa + xa; y = d[:, 1] * ha + ya; w = np.exp(d[:, 2]) * wa; h = np.exp(d[:, 3]) * ha
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 1)
''',
tests='''
def test_iou_values():
    a = np.array([[0, 0, 2, 2]]); b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]])
    assert np.allclose(box_iou(a, b), [[1 / 7, 1.0, 0.0]])

def test_giou_bounds_and_separated():
    a = np.array([[0, 0, 1, 1], [0, 0, 2, 2]]); b = np.array([[2, 0, 3, 1], [1, 1, 3, 3]])
    g = generalized_iou(a, b)
    assert np.isclose(g[0], 0 - (3 - 2) / 3)
    assert np.isclose(g[1], 1 / 7 - (9 - 7) / 9)
    assert np.all(g <= box_iou(a, b).diagonal() + 1e-12)

def test_roundtrip():
    rng = np.random.default_rng(0)
    anchors = np.concatenate([rng.random((20, 2)) * 50, rng.random((20, 2)) * 50 + 60], 1)
    boxes = np.concatenate([rng.random((20, 2)) * 40, rng.random((20, 2)) * 40 + 55], 1)
    assert np.allclose(decode(encode(boxes, anchors), anchors), boxes)
    assert np.allclose(encode(anchors, anchors), 0)
'''))

T.append(dict(
id="w13b_roi_align", title="RoI Align with bilinear sampling",
prompt="""Implement `roi_align(feature, box, out_size, sampling_ratio=2)` for a single-channel feature map (H, W),
following the `aligned=True` convention of torchvision:

- Pixel (i, j) has its value at continuous coordinates (x = j, y = i); shift the box by −0.5 so that box
  `(x1, y1, x2, y2)` in "pixel-edge" coordinates maps onto these centers.
- Split the box into `out_size × out_size` bins; in each bin take `sampling_ratio × sampling_ratio` evenly spaced
  sample points (at the centers of a regular sub-grid), bilinearly interpolate the feature at each, and average.
- Sample points outside `[−1, H] × [−1, W]` contribute 0; inside that range, clamp coordinates to the image.""",
starter='''import numpy as np

def bilinear(feature, y, x) -> float:
    raise NotImplementedError

def roi_align(feature, box, out_size: int, sampling_ratio: int = 2):
    raise NotImplementedError
''',
solution='''import numpy as np

def bilinear(feature, y, x):
    H, W = feature.shape
    if y < -1.0 or y > H or x < -1.0 or x > W:
        return 0.0
    y = min(max(y, 0.0), H - 1); x = min(max(x, 0.0), W - 1)
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = min(y0 + 1, H - 1), min(x0 + 1, W - 1)
    ly, lx = y - y0, x - x0
    return float((1 - ly) * (1 - lx) * feature[y0, x0] + (1 - ly) * lx * feature[y0, x1]
                 + ly * (1 - lx) * feature[y1, x0] + ly * lx * feature[y1, x1])

def roi_align(feature, box, out_size, sampling_ratio=2):
    x1, y1, x2, y2 = [float(v) - 0.5 for v in box]
    bh, bw = (y2 - y1) / out_size, (x2 - x1) / out_size
    out = np.zeros((out_size, out_size))
    s = sampling_ratio
    for i in range(out_size):
        for j in range(out_size):
            acc = 0.0
            for a in range(s):
                for b in range(s):
                    yy = y1 + i * bh + (a + 0.5) * bh / s
                    xx = x1 + j * bw + (b + 0.5) * bw / s
                    acc += bilinear(feature, yy, xx)
            out[i, j] = acc / (s * s)
    return out
''',
tests='''
def test_constant_feature():
    f = np.full((10, 10), 4.0)
    assert np.allclose(roi_align(f, (2, 2, 7, 8), 3), 4.0)

def test_linear_ramp_is_exact():
    H, W = 12, 14
    yy, xx = np.mgrid[0:H, 0:W]
    f = 2.0 * xx - 0.5 * yy + 1.0
    box = (2.0, 3.0, 9.0, 8.0); n = 4
    out = roi_align(f, box, n, sampling_ratio=2)
    x1, y1, x2, y2 = [v - 0.5 for v in box]
    cx = x1 + (np.arange(n) + 0.5) * (x2 - x1) / n
    cy = y1 + (np.arange(n) + 0.5) * (y2 - y1) / n
    assert np.allclose(out, 2.0 * cx[None, :] - 0.5 * cy[:, None] + 1.0)

def test_bilinear_corners_and_outside():
    f = np.arange(12.0).reshape(3, 4)
    assert bilinear(f, 1, 2) == f[1, 2]
    assert np.isclose(bilinear(f, 0.5, 0.5), f[:2, :2].mean())
    assert bilinear(f, -2, 0) == 0.0
'''))

T.append(dict(
id="w09_loss_zoo", title="Loss zoo: focal, Dice, pinball, label smoothing",
prompt="""Implement four losses (all return a scalar mean unless stated):

- `focal_loss(logits, targets, alpha=0.25, gamma=2.0)`: binary focal loss on logits with targets in {0, 1}:
  `−α_t (1 − p_t)^γ log p_t`, where `p_t = p` for positives and `1 − p` for negatives, `α_t = α` for positives and
  `1 − α` for negatives. Compute `log p_t` stably from logits.
- `dice_loss(probs, targets, eps=1.0)`: soft Dice over all elements, `1 − (2Σpt + eps) / (Σp + Σt + eps)`.
- `pinball_loss(y, pred, tau)`: mean of `max(τ(y − pred), (τ − 1)(y − pred))`.
- `label_smoothing_ce(logits, y, eps=0.1)`: multiclass CE against targets `(1 − eps)·onehot + eps/C`.""",
starter='''import numpy as np

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    raise NotImplementedError

def dice_loss(probs, targets, eps=1.0):
    raise NotImplementedError

def pinball_loss(y, pred, tau):
    raise NotImplementedError

def label_smoothing_ce(logits, y, eps=0.1):
    raise NotImplementedError
''',
solution='''import numpy as np

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    z = np.asarray(logits, float); t = np.asarray(targets, float)
    logp = -np.logaddexp(0.0, -z); log1mp = -np.logaddexp(0.0, z)
    log_pt = t * logp + (1 - t) * log1mp
    pt = np.exp(log_pt)
    at = t * alpha + (1 - t) * (1 - alpha)
    return float(np.mean(-at * (1 - pt) ** gamma * log_pt))

def dice_loss(probs, targets, eps=1.0):
    p = np.asarray(probs, float).ravel(); t = np.asarray(targets, float).ravel()
    return float(1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps))

def pinball_loss(y, pred, tau):
    r = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(tau * r, (tau - 1) * r)))

def label_smoothing_ce(logits, y, eps=0.1):
    z = np.asarray(logits, float); z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    C = z.shape[1]
    q = np.full_like(logp, eps / C); q[np.arange(len(y)), y] += 1 - eps
    return float(-(q * logp).sum(1).mean())
''',
tests='''
def _bce(z, t):
    return np.mean(np.logaddexp(0, z) - t * z)

def test_focal_reduces_to_bce():
    rng = np.random.default_rng(0)
    z = rng.normal(size=50); t = (rng.random(50) > 0.5).astype(float)
    assert np.isclose(focal_loss(z, t, alpha=0.5, gamma=0.0), 0.5 * _bce(z, t))

def test_focal_downweights_easy_and_stable():
    easy = focal_loss(np.array([8.0]), np.array([1.0])); hard = focal_loss(np.array([-2.0]), np.array([1.0]))
    assert easy < 1e-6 < hard
    assert np.isfinite(focal_loss(np.array([1e4, -1e4]), np.array([0.0, 1.0])))

def test_dice():
    t = np.array([[0, 1], [1, 1]], float)
    assert np.isclose(dice_loss(t, t, eps=0.0), 0.0)
    assert np.isclose(dice_loss(1 - t, t, eps=0.0), 1.0)

def test_pinball_minimizer_is_quantile():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2001)
    grid = np.linspace(-3, 3, 1201)
    best = grid[np.argmin([pinball_loss(y, c, 0.9) for c in grid])]
    assert abs(best - np.quantile(y, 0.9)) < 0.02

def test_label_smoothing():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(6, 4)); y = rng.integers(0, 4, 6)
    zs = z - z.max(1, keepdims=True); logp = zs - np.log(np.exp(zs).sum(1, keepdims=True))
    ce = -logp[np.arange(6), y].mean()
    assert np.isclose(label_smoothing_ce(z, y, 0.0), ce)
    assert np.isclose(label_smoothing_ce(z, y, 0.2), 0.8 * ce + 0.2 * (-logp.mean(1)).mean())
'''))

T.append(dict(
id="w13b_average_precision", title="Average precision for object detection",
prompt="""Implement `average_precision(detections, ground_truth, iou_thr=0.5)` for a single class.

- `detections`: list of `(image_id, score, box)`; `ground_truth`: dict `image_id → array (G, 4)` of xyxy boxes.
- Sort detections by score (descending). Each detection matches the unmatched ground-truth box in its image with the
  highest IoU, if that IoU ≥ `iou_thr` (true positive); otherwise it is a false positive. A ground-truth box can be
  matched only once.
- Build the precision–recall curve, make precision monotonically non-increasing from the right (the envelope), and
  return the area under it using all recall change points (VOC 2010+ / "all-point" interpolation).
Return 0.0 if there are no ground-truth boxes.""",
starter='''import numpy as np

def average_precision(detections, ground_truth, iou_thr=0.5) -> float:
    raise NotImplementedError
''',
solution='''import numpy as np

def _iou(a, B):
    lt = np.maximum(a[:2], B[:, :2]); rb = np.minimum(a[2:], B[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(a) + area(B) - inter)

def average_precision(detections, ground_truth, iou_thr=0.5):
    n_gt = sum(len(v) for v in ground_truth.values())
    if n_gt == 0:
        return 0.0
    used = {k: np.zeros(len(v), bool) for k, v in ground_truth.items()}
    dets = sorted(detections, key=lambda d: -d[1])
    tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
    for i, (img, _, box) in enumerate(dets):
        gts = np.asarray(ground_truth.get(img, np.zeros((0, 4))), float)
        if len(gts):
            ious = _iou(np.asarray(box, float), gts)
            ious[used[img]] = -1
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                tp[i] = 1; used[img][j] = True
                continue
        fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
''',
tests='''
A = [0, 0, 10, 10]; B = [20, 20, 30, 30]; FAR = [50, 50, 60, 60]

def test_perfect():
    gt = {0: np.array([A]), 1: np.array([B])}
    assert average_precision([(0, 0.9, A), (1, 0.8, B)], gt) == 1.0

def test_hand_example():
    gt = {0: np.array([A]), 1: np.array([B])}
    dets = [(0, 0.9, A), (1, 0.8, FAR), (1, 0.7, B)]
    assert np.isclose(average_precision(dets, gt), 0.5 * 1.0 + 0.5 * (2 / 3))

def test_duplicates_are_false_positives():
    gt = {0: np.array([A])}
    dets = [(0, 0.9, A), (0, 0.8, [0, 0, 10, 9.5])]
    assert np.isclose(average_precision(dets, gt), 1.0)
    assert np.isclose(average_precision([(0, 0.9, [0, 0, 10, 9.5]), (0, 0.95, FAR)], gt), 0.5)

def test_iou_threshold():
    gt = {0: np.array([A])}
    shifted = [4, 0, 14, 10]   # IoU = 6/14
    assert average_precision([(0, 0.9, shifted)], gt, 0.5) == 0.0
    assert average_precision([(0, 0.9, shifted)], gt, 0.4) == 1.0

def test_no_ground_truth():
    assert average_precision([(0, 0.5, A)], {}) == 0.0
'''))
