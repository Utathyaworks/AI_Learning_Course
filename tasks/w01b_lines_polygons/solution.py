import numpy as np

def pairwise_dist(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.clip(d2, 0, None))

def hyperplane_signed_distance(X, w, b):
    w = np.asarray(w, float)
    return (np.asarray(X, float) @ w + b) / np.linalg.norm(w)

def point_line_distance(P, a, b):
    P = np.asarray(P, float); a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    cross = d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])
    return np.abs(cross) / np.linalg.norm(d)

def polygon_area(V):
    V = np.asarray(V, float)
    x, y = V[:, 0], V[:, 1]
    return float(0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1)))

def convex_hull(P):
    pts = sorted(set(map(tuple, np.asarray(P, float))))
    if len(pts) <= 2:
        return np.array(pts)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])
