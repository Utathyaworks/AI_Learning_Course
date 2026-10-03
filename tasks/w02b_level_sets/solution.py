import numpy as np

def grid_eval(f, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    return f(X, Y)

def grid_argmin(Z, xs, ys):
    i, j = np.unravel_index(np.argmin(Z), Z.shape)
    return float(xs[j]), float(ys[i])

def quadratic_shape(A):
    lam = np.linalg.eigvalsh(np.asarray(A, float))
    if np.any(np.abs(lam) < 1e-12):
        return "trough"
    if np.all(lam > 0):
        return "bowl"
    if np.all(lam < 0):
        return "cap"
    return "saddle"

def level_set_ellipse_axes(A, c):
    lam = np.linalg.eigvalsh(np.asarray(A, float))
    return np.sort(np.sqrt(c / lam))[::-1]

def contour_crossings(z_line, level):
    s = np.sign(np.asarray(z_line, float) - level)
    return np.nonzero(s[:-1] * s[1:] < 0)[0]
