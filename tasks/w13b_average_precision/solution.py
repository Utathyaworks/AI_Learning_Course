import numpy as np

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
