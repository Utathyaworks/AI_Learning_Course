import numpy as np
from submission import *  # noqa: F401,F403
def test_first_step_size_is_lr():
    w = np.array([1.0, -2.0]); g = np.array([0.3, -5.0])
    w1, m, v = adam_step(w, g, np.zeros(2), np.zeros(2), 1, lr=0.01)
    assert np.allclose(w - w1, 0.01 * np.sign(g), atol=1e-6)

def test_matches_reference_sequence():
    rng = np.random.default_rng(0)
    w = rng.normal(size=3); m = np.zeros(3); v = np.zeros(3)
    wr, mr, vr = w.copy(), np.zeros(3), np.zeros(3)
    for t in range(1, 6):
        g = rng.normal(size=3)
        w, m, v = adam_step(w, g, m, v, t, lr=0.1, weight_decay=0.01)
        wr = wr - 0.1 * 0.01 * wr
        mr = 0.9 * mr + 0.1 * g; vr = 0.999 * vr + 0.001 * g * g
        wr = wr - 0.1 * (mr / (1 - 0.9 ** t)) / (np.sqrt(vr / (1 - 0.999 ** t)) + 1e-8)
    assert np.allclose(w, wr) and np.allclose(m, mr) and np.allclose(v, vr)

def test_adam_l2_differs_from_adamw():
    w = np.array([5.0]); g = np.array([0.1])
    a, *_ = adam_step(w, g, np.zeros(1), np.zeros(1), 1, lr=0.1, weight_decay=0.5, decoupled=True)
    b, *_ = adam_step(w, g, np.zeros(1), np.zeros(1), 1, lr=0.1, weight_decay=0.5, decoupled=False)
    assert abs(a[0] - (5.0 - 0.1 * 0.5 * 5.0 - 0.1)) < 1e-6
    assert abs(b[0] - (5.0 - 0.1)) < 1e-6
