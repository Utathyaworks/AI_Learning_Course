import numpy as np
from submission import *  # noqa: F401,F403
from scipy import ndimage

def test_kernel():
    k = gaussian_kernel1d(1.5)
    assert len(k) == 2 * 5 + 1 and np.isclose(k.sum(), 1) and np.allclose(k, k[::-1])

def test_blur_equals_2d_correlation():
    rng = np.random.default_rng(0)
    img = rng.random((20, 17))
    k = gaussian_kernel1d(1.2)
    ref = ndimage.correlate(img, np.outer(k, k), mode="mirror")
    assert np.allclose(gaussian_blur(img, 1.2), ref)

def test_blur_keeps_mean_of_constant():
    assert np.allclose(gaussian_blur(np.full((8, 8), 3.0), 2.0), 3.0)

def test_sobel_matches_scipy():
    rng = np.random.default_rng(1)
    img = rng.random((15, 12))
    gx, gy, mag = sobel(img)
    assert np.allclose(gx, ndimage.sobel(img, axis=1, mode="reflect"))
    assert np.allclose(gy, ndimage.sobel(img, axis=0, mode="reflect"))
    assert np.allclose(mag, np.hypot(gx, gy))

def test_step_edge_direction():
    img = np.zeros((9, 9)); img[:, 5:] = 1
    gx, gy, _ = sobel(img)
    assert gx[4, 4] > 0 and np.allclose(gy, 0)
