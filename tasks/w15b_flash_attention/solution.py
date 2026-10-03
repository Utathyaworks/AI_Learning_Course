import numpy as np

def attention(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    if causal:
        S = np.where(np.tril(np.ones(S.shape, bool)), S, -np.inf)
    S = S - S.max(1, keepdims=True)
    P = np.exp(S); P /= P.sum(1, keepdims=True)
    return P @ V

def tiled_attention(Q, K, V, block=16):
    n, d = Q.shape
    m_run = np.full(n, -np.inf); l_run = np.zeros(n); o = np.zeros((n, V.shape[1]))
    for s in range(0, K.shape[0], block):
        S = Q @ K[s:s + block].T / np.sqrt(d)
        m_new = np.maximum(m_run, S.max(1))
        scale = np.exp(m_run - m_new)
        P = np.exp(S - m_new[:, None])
        l_run = l_run * scale + P.sum(1)
        o = o * scale[:, None] + P @ V[s:s + block]
        m_run = m_new
    return o / l_run[:, None]

def decode_with_cache(Q, K, V):
    kc, vc, out = [], [], []
    for t in range(Q.shape[0]):
        kc.append(K[t]); vc.append(V[t])
        out.append(attention(Q[t:t + 1], np.array(kc), np.array(vc))[0])
    return np.array(out)

def gqa_expand(K, n_q_heads):
    g = n_q_heads // K.shape[0]
    return np.repeat(K, g, axis=0)
