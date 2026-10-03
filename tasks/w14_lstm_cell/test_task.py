import numpy as np
from submission import *  # noqa: F401,F403
def test_shapes_and_reference():
    rng = np.random.default_rng(0)
    N, D, H = 3, 4, 5
    x, h, c = rng.normal(size=(N, D)), rng.normal(size=(N, H)), rng.normal(size=(N, H))
    W, U, b = rng.normal(size=(D, 4*H)), rng.normal(size=(H, 4*H)), rng.normal(size=4*H)
    h2, c2 = lstm_cell(x, h, c, W, U, b)
    s = lambda z: 1 / (1 + np.exp(-z))
    a = x @ W + h @ U + b
    ci = s(a[:, H:2*H]) * c + s(a[:, :H]) * np.tanh(a[:, 2*H:3*H])
    assert np.allclose(c2, ci) and np.allclose(h2, s(a[:, 3*H:]) * np.tanh(ci))

def test_forget_gate_keeps_memory():
    H = 2
    W = np.zeros((1, 4*H)); U = np.zeros((H, 4*H))
    b = np.concatenate([np.full(H, -50.0), np.full(H, 50.0), np.zeros(H), np.full(H, 50.0)])
    c = np.array([[0.3, -0.7]])
    h2, c2 = lstm_cell(np.zeros((1, 1)), np.zeros((1, H)), c, W, U, b)
    assert np.allclose(c2, c) and np.allclose(h2, np.tanh(c))
