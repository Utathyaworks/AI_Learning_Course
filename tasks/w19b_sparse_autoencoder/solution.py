import numpy as np

def normalize_decoder(W_dec):
    W = np.asarray(W_dec, float)
    return W / np.linalg.norm(W, axis=1, keepdims=True)

def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    f = np.maximum((X - b_dec) @ W_enc + b_enc, 0.0)
    return f, f @ W_dec + b_dec

def sae_loss(X, f, X_hat, l1):
    return float(np.mean(np.sum((X - X_hat) ** 2, 1) + l1 * np.sum(np.abs(f), 1)))

def topk_encode(pre, k):
    pre = np.maximum(np.asarray(pre, float), 0.0)
    out = np.zeros_like(pre)
    idx = np.argsort(-pre, axis=1)[:, :k]
    np.put_along_axis(out, idx, np.take_along_axis(pre, idx, 1), 1)
    return out

def sae_metrics(X, f, X_hat):
    l0 = float(np.mean(np.sum(f > 0, 1)))
    fve = float(1 - np.sum((X - X_hat) ** 2) / np.sum((X - X.mean(0)) ** 2))
    dead = float(np.mean(~np.any(f > 0, 0)))
    return {"l0": l0, "fve": fve, "dead": dead}
