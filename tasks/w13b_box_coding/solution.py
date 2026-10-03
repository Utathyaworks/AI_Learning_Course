import numpy as np

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
