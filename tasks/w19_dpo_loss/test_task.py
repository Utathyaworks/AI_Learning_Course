import numpy as np
from submission import *  # noqa: F401,F403
def test_equal_policies_log2():
    z = np.zeros(4)
    loss, acc, gc, gr = dpo_loss(z, z, z, z, 0.1)
    assert np.isclose(loss, np.log(2)) and acc == 0.0
    assert np.allclose(gc, -0.1 * 0.5 / 4) and np.allclose(gr, 0.1 * 0.5 / 4)

def test_numeric_gradient_and_acc():
    rng = np.random.default_rng(0)
    pc, pr, rc, rr = (rng.normal(size=5) for _ in range(4))
    loss, acc, gc, gr = dpo_loss(pc, pr, rc, rr, 0.5)
    for i in range(5):
        e = np.zeros(5); e[i] = 1e-6
        assert abs((dpo_loss(pc + e, pr, rc, rr, 0.5)[0] - dpo_loss(pc - e, pr, rc, rr, 0.5)[0]) / 2e-6 - gc[i]) < 1e-6
        assert abs((dpo_loss(pc, pr + e, rc, rr, 0.5)[0] - dpo_loss(pc, pr - e, rc, rr, 0.5)[0]) / 2e-6 - gr[i]) < 1e-6
    assert acc == np.mean(0.5 * (pc - rc) > 0.5 * (pr - rr))

def test_stable_large_margin():
    loss, *_ = dpo_loss(np.array([-1e4]), np.array([1e4]), np.zeros(1), np.zeros(1), 1.0)
    assert np.isfinite(loss) and np.isclose(loss, 2e4)
