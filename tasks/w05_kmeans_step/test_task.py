import numpy as np
from submission import *  # noqa: F401,F403
def test_two_blobs():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [10.0, 10.0], [10.0, 11.0]])
    C = np.array([[1.0, 1.0], [9.0, 9.0]])
    lab, new, inert = kmeans_step(X, C)
    assert list(lab) == [0, 0, 1, 1]
    assert np.allclose(new, [[0.0, 0.5], [10.0, 10.5]])
    assert abs(inert - (2 + 1 + 2 + 5)) < 1e-9

def test_empty_cluster_kept():
    X = np.array([[0.0], [1.0]]); C = np.array([[0.5], [100.0]])
    lab, new, _ = kmeans_step(X, C)
    assert new[1, 0] == 100.0

def test_objective_never_increases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3)); C = X[:4].copy()
    prev = np.inf
    for _ in range(10):
        lab, C, inert = kmeans_step(X, C)
        assert inert <= prev + 1e-9
        prev = inert
