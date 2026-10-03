import numpy as np
from submission import *  # noqa: F401,F403
def test_two_state_closed_form():
    # state 0: action 0 stays (+1), action 1 goes to 1 (+0); state 1 absorbing with +2
    P = np.zeros((2, 2, 2)); P[0, 0, 0] = 1; P[0, 1, 1] = 1; P[1, :, 1] = 1
    R = np.array([[1.0, 0.0], [2.0, 2.0]])
    V, pi = value_iteration(P, R, 0.9)
    assert np.allclose(V[1], 2 / 0.1) and np.allclose(V[0], 0.9 * 20)
    assert pi[0] == 1

def test_bellman_fixed_point_random():
    rng = np.random.default_rng(0)
    S, A = 6, 3
    P = rng.random((S, A, S)); P /= P.sum(-1, keepdims=True)
    R = rng.normal(size=(S, A))
    V, pi = value_iteration(P, R, 0.95)
    assert np.allclose(V, (R + 0.95 * P @ V).max(1), atol=1e-8)
    Ppi = P[np.arange(S), pi]; Rpi = R[np.arange(S), pi]
    assert np.allclose(V, np.linalg.solve(np.eye(S) - 0.95 * Ppi, Rpi), atol=1e-6)
