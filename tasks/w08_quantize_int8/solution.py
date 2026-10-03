import numpy as np

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
