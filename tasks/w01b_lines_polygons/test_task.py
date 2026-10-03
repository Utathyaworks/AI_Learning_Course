import numpy as np
from submission import *  # noqa: F401,F403
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

def test_pairwise():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(30, 5)), rng.normal(size=(20, 5))
    assert np.allclose(pairwise_dist(X, Y), cdist(X, Y), atol=1e-8)
    assert np.allclose(np.diag(pairwise_dist(X, X)), 0, atol=1e-6)

def test_hyperplane():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [-3.0, -4.0]])
    w, b = np.array([3.0, 4.0]), -5.0
    assert np.allclose(hyperplane_signed_distance(X, w, b), [4.0, -1.0, -6.0])
    assert np.allclose(hyperplane_signed_distance(X, 10 * w, 10 * b), [4.0, -1.0, -6.0])

def test_point_line():
    P = np.array([[0.0, 1.0], [5.0, -2.0], [2.0, 2.0]])
    assert np.allclose(point_line_distance(P, [0, 0], [1, 0]), [1, 2, 2])
    assert np.allclose(point_line_distance(np.array([[0.0, 0.0]]), [1, 0], [0, 1]), [np.sqrt(0.5)])

def test_area_sign():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert polygon_area(sq) == 4.0 and polygon_area(sq[::-1]) == -4.0
    tri = np.array([[0, 0], [4, 0], [0, 3]])
    assert polygon_area(tri) == 6.0

def test_hull():
    rng = np.random.default_rng(1)
    P = rng.random((200, 2))
    h = convex_hull(P)
    ref = P[ConvexHull(P).vertices]
    assert len(h) == len(ref)
    assert {tuple(np.round(p, 12)) for p in h} == {tuple(np.round(p, 12)) for p in ref}
    assert polygon_area(h) > 0
    sq = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2], [1, 1]], float)
    assert np.array_equal(convex_hull(sq), [[0, 0], [2, 0], [2, 2], [0, 2]])
