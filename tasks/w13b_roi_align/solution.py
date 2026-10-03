import numpy as np

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
