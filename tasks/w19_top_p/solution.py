import numpy as np

def sampling_probs(logits, temperature=1.0, top_p=1.0):
    z = np.asarray(logits, float) / temperature
    z = z - z.max()
    p = np.exp(z); p /= p.sum()
    order = np.lexsort((np.arange(len(p)), -p))
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, top_p - 1e-12) + 1)
    keep = order[:max(1, k)]
    out = np.zeros_like(p); out[keep] = p[keep]
    return out / out.sum()
