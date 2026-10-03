import numpy as np
from submission import *  # noqa: F401,F403
import warnings

def _num(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        g[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return g

def test_gradient_matches_numeric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3)); y = (rng.random(20) > 0.5).astype(float)
    w = rng.normal(size=3); b = 0.3
    loss, dw, db = logistic_loss_grad(X, y, w, b, l2=0.1)
    assert np.allclose(dw, _num(lambda v: logistic_loss_grad(X, y, v, b, 0.1)[0], w), atol=1e-6)
    assert abs(db - (logistic_loss_grad(X, y, w, b + 1e-6, 0.1)[0] - logistic_loss_grad(X, y, w, b - 1e-6, 0.1)[0]) / 2e-6) < 1e-6

def test_known_value():
    X = np.zeros((2, 1)); y = np.array([0.0, 1.0])
    loss, dw, db = logistic_loss_grad(X, y, np.zeros(1), 0.0)
    assert abs(loss - np.log(2)) < 1e-12 and abs(db) < 1e-12

def test_stable_for_huge_logits():
    X = np.array([[1000.0], [-1000.0]]); y = np.array([0.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss, dw, db = logistic_loss_grad(X, y, np.ones(1), 0.0)
    assert np.isfinite(loss) and abs(loss - 1000.0) < 1e-6
    assert np.all(np.isfinite(dw))
