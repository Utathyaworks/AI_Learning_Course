import numpy as np

def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    ln = np.asarray(logp_new, float); lo = np.asarray(logp_old, float); A = np.asarray(adv, float)
    rho = np.exp(ln - lo)
    unclipped = rho * A
    clipped = np.clip(rho, 1 - eps, 1 + eps) * A
    use_unclipped = unclipped <= clipped
    obj = np.where(use_unclipped, unclipped, clipped)
    n = len(A)
    grad = np.where(use_unclipped, -rho * A / n, 0.0)
    clip_frac = float(np.mean(~use_unclipped))
    return float(-obj.mean()), grad, clip_frac
