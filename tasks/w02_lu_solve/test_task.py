import numpy as np
from submission import *  # noqa: F401,F403
def test_random_vector():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)); b = rng.normal(size=6)
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b), atol=1e-8)

def test_needs_pivoting():
    A = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0], [2.0, 0.0, 3.0]])
    b = np.array([1.0, 2.0, 3.0])
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b))

def test_matrix_rhs():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); B = rng.normal(size=(5, 3))
    x = lu_solve(A, B)
    assert x.shape == (5, 3)
    assert np.allclose(A @ x, B, atol=1e-8)

def test_small_pivot_stability():
    A = np.array([[1e-12, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    assert np.allclose(lu_solve(A, b), np.linalg.solve(A, b), atol=1e-8)
