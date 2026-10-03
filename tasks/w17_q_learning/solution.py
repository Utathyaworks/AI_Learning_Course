import numpy as np

def q_update(Q, s, a, r, s2, done, alpha, gamma):
    Q = np.array(Q, float)
    target = r + (0.0 if done else gamma * Q[s2].max())
    Q[s, a] += alpha * (target - Q[s, a])
    return Q

def double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma):
    QA = np.array(QA, float)
    if done:
        target = r
    else:
        a_star = int(np.argmax(QA[s2]))
        target = r + gamma * QB[s2, a_star]
    QA[s, a] += alpha * (target - QA[s, a])
    return QA
