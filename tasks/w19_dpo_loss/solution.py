import numpy as np

def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
    pc, pr = np.asarray(pi_chosen, float), np.asarray(pi_rejected, float)
    rc = beta * (pc - np.asarray(ref_chosen, float))
    rr = beta * (pr - np.asarray(ref_rejected, float))
    m = rc - rr
    loss = np.mean(np.logaddexp(0.0, -m))
    s = np.exp(-np.logaddexp(0.0, m))  # sigma(-m)
    n = len(m)
    return float(loss), float(np.mean(rc > rr)), -beta * s / n, beta * s / n
