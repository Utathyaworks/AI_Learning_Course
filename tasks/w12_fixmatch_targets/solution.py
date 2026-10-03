import numpy as np

def fixmatch_unlabeled_loss(probs_weak, logits_strong, threshold=0.95):
    p = np.asarray(probs_weak, float)
    z = np.asarray(logits_strong, float)
    pl = p.argmax(1)
    mask = p.max(1) >= threshold
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    ce = -logp[np.arange(len(pl)), pl]
    return float((ce * mask).sum() / len(pl)), mask, pl
