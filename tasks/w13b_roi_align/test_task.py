import numpy as np
from submission import *  # noqa: F401,F403
def test_constant_feature():
    f = np.full((10, 10), 4.0)
    assert np.allclose(roi_align(f, (2, 2, 7, 8), 3), 4.0)

def test_linear_ramp_is_exact():
    H, W = 12, 14
    yy, xx = np.mgrid[0:H, 0:W]
    f = 2.0 * xx - 0.5 * yy + 1.0
    box = (2.0, 3.0, 9.0, 8.0); n = 4
    out = roi_align(f, box, n, sampling_ratio=2)
    x1, y1, x2, y2 = [v - 0.5 for v in box]
    cx = x1 + (np.arange(n) + 0.5) * (x2 - x1) / n
    cy = y1 + (np.arange(n) + 0.5) * (y2 - y1) / n
    assert np.allclose(out, 2.0 * cx[None, :] - 0.5 * cy[:, None] + 1.0)

def test_bilinear_corners_and_outside():
    f = np.arange(12.0).reshape(3, 4)
    assert bilinear(f, 1, 2) == f[1, 2]
    assert np.isclose(bilinear(f, 0.5, 0.5), f[:2, :2].mean())
    assert bilinear(f, -2, 0) == 0.0
