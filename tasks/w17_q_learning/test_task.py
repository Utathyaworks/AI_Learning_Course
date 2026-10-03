import numpy as np
from submission import *  # noqa: F401,F403
def test_q_update_value_and_no_mutation():
    Q = np.zeros((3, 2)); Q[2] = [1.0, 5.0]; Q0 = Q.copy()
    Q2 = q_update(Q, 0, 1, 1.0, 2, False, 0.5, 0.9)
    assert np.isclose(Q2[0, 1], 0.5 * (1 + 0.9 * 5))
    assert np.array_equal(Q, Q0)

def test_terminal():
    Q = np.ones((2, 2)) * 10
    assert np.isclose(q_update(Q, 0, 0, 1.0, 1, True, 1.0, 0.9)[0, 0], 1.0)

def test_double_q_uses_other_table():
    QA = np.zeros((2, 2)); QA[1] = [3.0, 1.0]
    QB = np.zeros((2, 2)); QB[1] = [-2.0, 7.0]
    new = double_q_update(QA, QB, 0, 0, 0.0, 1, False, 1.0, 1.0)
    assert np.isclose(new[0, 0], -2.0)

def test_converges_on_bandit():
    rng = np.random.default_rng(0)
    Q = np.zeros((1, 2))
    for _ in range(5000):
        a = rng.integers(2); r = rng.normal(1.0 if a else 0.0, 0.1)
        Q = q_update(Q, 0, a, r, 0, True, 0.01, 0.0)
    assert abs(Q[0, 1] - 1.0) < 0.05 and abs(Q[0, 0]) < 0.05
