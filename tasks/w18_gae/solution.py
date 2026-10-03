import numpy as np

def gae(rewards, values, dones, last_value, gamma, lam):
    r = np.asarray(rewards, float); v = np.asarray(values, float); d = np.asarray(dones, float)
    T = len(r)
    adv = np.zeros(T)
    nxt_v, nxt_a = last_value, 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - d[t]
        delta = r[t] + gamma * nxt_v * nonterm - v[t]
        nxt_a = delta + gamma * lam * nonterm * nxt_a
        adv[t] = nxt_a
        nxt_v = v[t]
    return adv, adv + v
