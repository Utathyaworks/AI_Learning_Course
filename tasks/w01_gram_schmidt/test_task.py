import numpy as np
from submission import *  # noqa: F401,F403
def test_orthonormal():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 5))
    Q = gram_schmidt(A)
    assert Q.shape == (8, 5)
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-8)

def test_same_span_and_positive_r():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 4))
    Q = gram_schmidt(A)
    R = Q.T @ A
    assert np.allclose(np.tril(R, -1), 0, atol=1e-8)
    assert np.all(np.diag(R) > 0)
    assert np.allclose(Q @ R, A, atol=1e-8)

def test_matches_qr_up_to_sign():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(5, 5))
    Q = gram_schmidt(A)
    Q2, R2 = np.linalg.qr(A)
    Q2 = Q2 * np.sign(np.diag(R2))
    assert np.allclose(Q, Q2, atol=1e-8)
