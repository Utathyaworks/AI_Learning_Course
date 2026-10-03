import numpy as np

def forward(tokens, W, patch=None):
    tokens = np.asarray(tokens); n = len(tokens)
    x0 = W["E"][tokens] + W["P"][:n]
    q, k = x0 @ W["WQ"], x0 @ W["WK"]
    S = q @ k.T / np.sqrt(q.shape[1])
    S = np.where(np.tril(np.ones((n, n), bool)), S, -np.inf)
    S = S - S.max(1, keepdims=True); A = np.exp(S); A /= A.sum(1, keepdims=True)
    head = A @ (x0 @ W["WV"]) @ W["WO"]
    if patch is not None and "head" in patch:
        head = patch["head"]
    logits = (x0 + head) @ W["U"]
    return logits, {"x0": x0, "A": A, "head": head}

def qk_circuit(W):
    return (W["E"] @ W["WQ"]) @ (W["E"] @ W["WK"]).T

def ov_circuit(W):
    return W["E"] @ W["WV"] @ W["WO"] @ W["U"]

def direct_logit_attribution(cache, W, pos, token):
    return float(cache["x0"][pos] @ W["U"][:, token]), float(cache["head"][pos] @ W["U"][:, token])

def patching_effect(clean, corrupt, W, pos, token):
    lc, cc = forward(clean, W)
    lx, _ = forward(corrupt, W)
    lp, _ = forward(corrupt, W, patch={"head": cc["head"]})
    return float((lp[pos, token] - lx[pos, token]) / (lc[pos, token] - lx[pos, token]))
