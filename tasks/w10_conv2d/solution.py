import numpy as np

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
