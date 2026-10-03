import numpy as np

def _ce_and_dlogits(L):
    Z = L - L.max(1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    n = len(L)
    loss = -np.mean(np.diag(logp))
    g = np.exp(logp); g[np.arange(n), np.arange(n)] -= 1
    return loss, g / n

def clip_loss(img, txt, logit_scale):
    a = img / np.linalg.norm(img, axis=1, keepdims=True)
    b = txt / np.linalg.norm(txt, axis=1, keepdims=True)
    S = a @ b.T
    s = np.exp(logit_scale)
    L = s * S
    l1, g1 = _ce_and_dlogits(L)
    l2, g2 = _ce_and_dlogits(L.T)
    loss = 0.5 * (l1 + l2)
    dL = 0.5 * (g1 + g2.T)
    return float(loss), float((dL * S).sum() * s)
