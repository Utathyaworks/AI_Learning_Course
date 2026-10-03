import numpy as np

def ctc_loss(log_probs, target, blank=0):
    lp = np.asarray(log_probs, float)
    T = lp.shape[0]
    ext = [blank]
    for c in target:
        ext += [c, blank]
    S = len(ext)
    alpha = np.full(S, -np.inf)
    alpha[0] = lp[0, ext[0]]
    if S > 1:
        alpha[1] = lp[0, ext[1]]
    for t in range(1, T):
        new = np.full(S, -np.inf)
        for s in range(S):
            terms = [alpha[s]]
            if s >= 1:
                terms.append(alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[s - 2])
            new[s] = np.logaddexp.reduce(terms) + lp[t, ext[s]]
        alpha = new
    ends = [alpha[S - 1]] + ([alpha[S - 2]] if S > 1 else [])
    ll = np.logaddexp.reduce(ends)
    return float(-ll) if np.isfinite(ll) else float("inf")
