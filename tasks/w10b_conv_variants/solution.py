import numpy as np

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
