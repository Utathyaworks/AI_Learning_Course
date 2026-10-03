import numpy as np

def value_iteration(P, R, gamma, tol=1e-10):
    V = np.zeros(P.shape[0])
    while True:
        Q = R + gamma * P @ V
        V2 = Q.max(1)
        if np.max(np.abs(V2 - V)) < tol:
            V = V2
            break
        V = V2
    Q = R + gamma * P @ V
    return V, np.argmax(Q, axis=1)
