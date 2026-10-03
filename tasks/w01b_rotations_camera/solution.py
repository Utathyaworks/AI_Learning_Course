import numpy as np

def rodrigues(axis, angle):
    k = np.asarray(axis, float); k = k / np.linalg.norm(k)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx

def quat_to_matrix(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def quat_multiply(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return np.array([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                     a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                     a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                     a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])

def homogeneous(R, t):
    H = np.eye(4); H[:3, :3] = R; H[:3, 3] = np.asarray(t, float)
    return H

def project(X, K, R, t):
    Xc = np.asarray(X, float) @ np.asarray(R, float).T + np.asarray(t, float)
    x = Xc @ np.asarray(K, float).T
    return x[:, :2] / x[:, 2:3]
