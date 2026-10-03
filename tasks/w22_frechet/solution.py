import numpy as np

def _sqrtm_psd(S):
    w, U = np.linalg.eigh((S + S.T) / 2)
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T

def stats(features):
    f = np.asarray(features, float)
    return f.mean(0), np.cov(f, rowvar=False)

def frechet_distance(mu1, S1, mu2, S2):
    r1 = _sqrtm_psd(S1)
    mid = _sqrtm_psd(r1 @ S2 @ r1)
    d = np.asarray(mu1) - np.asarray(mu2)
    return float(d @ d + np.trace(S1 + S2 - 2 * mid))
