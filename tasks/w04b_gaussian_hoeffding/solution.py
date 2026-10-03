import numpy as np

def gaussian_condition(mu, S, idx_obs, x_obs):
    mu = np.asarray(mu, float); S = np.asarray(S, float)
    o = np.asarray(idx_obs); u = np.setdiff1d(np.arange(len(mu)), o)
    Soo, Suo, Suu = S[np.ix_(o, o)], S[np.ix_(u, o)], S[np.ix_(u, u)]
    A = np.linalg.solve(Soo, Suo.T).T
    mean = mu[u] + A @ (np.asarray(x_obs, float) - mu[o])
    cov = Suu - A @ Suo.T
    return mean, cov

def kl_gaussians(mu0, S0, mu1, S1):
    mu0, mu1 = np.asarray(mu0, float), np.asarray(mu1, float)
    S0, S1 = np.asarray(S0, float), np.asarray(S1, float)
    k = len(mu0); d = mu1 - mu0
    tr = np.trace(np.linalg.solve(S1, S0))
    quad = d @ np.linalg.solve(S1, d)
    _, ld1 = np.linalg.slogdet(S1); _, ld0 = np.linalg.slogdet(S0)
    return float(0.5 * (tr + quad - k + ld1 - ld0))

def hoeffding_n(eps, delta):
    return int(np.ceil(np.log(2 / delta) / (2 * eps ** 2)))

def hoeffding_halfwidth(n, delta):
    return float(np.sqrt(np.log(2 / delta) / (2 * n)))
