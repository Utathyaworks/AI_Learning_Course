import numpy as np
from submission import *  # noqa: F401,F403
from scipy.spatial.transform import Rotation

def test_rodrigues_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        ax = rng.normal(size=3); ang = rng.uniform(-3, 3)
        ref = Rotation.from_rotvec(ax / np.linalg.norm(ax) * ang).as_matrix()
        R = rodrigues(ax, ang)
        assert np.allclose(R, ref) and np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)

def test_z_rotation():
    assert np.allclose(rodrigues([0, 0, 2], np.pi / 2) @ [1, 0, 0], [0, 1, 0])

def test_quaternion():
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        ref = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        assert np.allclose(quat_to_matrix(q), ref)
        assert np.allclose(quat_to_matrix(-q), ref)          # double cover
        p = rng.normal(size=4); p /= np.linalg.norm(p); qn = q / np.linalg.norm(q)
        assert np.allclose(quat_to_matrix(quat_multiply(p, qn)), quat_to_matrix(p) @ quat_to_matrix(qn))

def test_homogeneous_composes():
    R1, R2 = rodrigues([1, 2, 3], 0.4), rodrigues([0, 1, 0], -1.1)
    t1, t2 = np.array([1.0, 2, 3]), np.array([-2.0, 0, 5])
    x = np.array([0.3, -0.7, 2.0, 1.0])
    lhs = homogeneous(R1, t1) @ homogeneous(R2, t2) @ x
    assert np.allclose(lhs[:3], R1 @ (R2 @ x[:3] + t2) + t1)

def test_projection():
    K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
    X = np.array([[0.0, 0, 5], [1, 0, 5], [0, 1, 10]])
    assert np.allclose(project(X, K, np.eye(3), np.zeros(3)), [[320, 240], [420, 240], [320, 290]])
    assert np.allclose(project(2 * X, K, np.eye(3), np.zeros(3)), project(X, K, np.eye(3), np.zeros(3)))
