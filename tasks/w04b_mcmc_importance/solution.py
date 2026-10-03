import numpy as np

def importance_estimate(f, logp, q_sample, q_logpdf, n, rng):
    x = q_sample(n, rng)
    lw = logp(x) - q_logpdf(x)
    w = np.exp(lw - lw.max())
    est = float(np.sum(w * f(x)) / w.sum())
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    return est, ess

def metropolis_hastings(logp, x0, n, step, rng):
    x = np.atleast_1d(np.asarray(x0, float)).copy()
    lp = logp(x)
    out = np.empty((n, x.size)); acc = 0
    for i in range(n):
        y = x + step * rng.normal(size=x.size)
        ly = logp(y)
        if np.log(rng.random()) < ly - lp:
            x, lp = y, ly; acc += 1
        out[i] = x
    return out, acc / n
