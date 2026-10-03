import numpy as np
from submission import *  # noqa: F401,F403
from scipy import ndimage

def test_output_sizes():
    assert conv_out_size(32, 3) == 30
    assert conv_out_size(32, 3, stride=2, padding=1) == 16
    assert conv_out_size(32, 3, dilation=2) == 28
    assert conv_out_size(7, 3, stride=2) == 3

def test_dilated_equals_sparse_kernel():
    rng = np.random.default_rng(0)
    x, w = rng.random((12, 11)), rng.random((3, 3))
    big = np.zeros((5, 5)); big[::2, ::2] = w
    assert np.allclose(dilated_conv2d(x, w, 2), strided_conv2d(x, big, 1))

def test_strided_matches_scipy():
    rng = np.random.default_rng(1)
    x, w = rng.random((10, 9)), rng.random((3, 3))
    full = ndimage.correlate(x, w, mode="constant")[1:-1, 1:-1]
    assert np.allclose(strided_conv2d(x, w, 2), full[::2, ::2])

def test_transpose_is_adjoint():
    rng = np.random.default_rng(2)
    for s in (1, 2, 3):
        x, w = rng.random((11, 10)), rng.random((3, 3))
        y = rng.random(strided_conv2d(x, w, s).shape)
        lhs = (strided_conv2d(x, w, s) * y).sum()
        rhs = (x * conv_transpose2d(y, w, s, x.shape)).sum()
        assert np.isclose(lhs, rhs)
