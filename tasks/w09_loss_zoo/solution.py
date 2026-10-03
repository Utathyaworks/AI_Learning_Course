import numpy as np

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
