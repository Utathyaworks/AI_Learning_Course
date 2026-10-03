import numpy as np

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
