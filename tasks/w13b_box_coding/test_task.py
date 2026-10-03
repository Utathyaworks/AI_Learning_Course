import numpy as np
from submission import *  # noqa: F401,F403
def test_iou_values():
    a = np.array([[0, 0, 2, 2]]); b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]])
    assert np.allclose(box_iou(a, b), [[1 / 7, 1.0, 0.0]])

def test_giou_bounds_and_separated():
    a = np.array([[0, 0, 1, 1], [0, 0, 2, 2]]); b = np.array([[2, 0, 3, 1], [1, 1, 3, 3]])
    g = generalized_iou(a, b)
    assert np.isclose(g[0], 0 - (3 - 2) / 3)
    assert np.isclose(g[1], 1 / 7 - (9 - 7) / 9)
    assert np.all(g <= box_iou(a, b).diagonal() + 1e-12)

def test_roundtrip():
    rng = np.random.default_rng(0)
    anchors = np.concatenate([rng.random((20, 2)) * 50, rng.random((20, 2)) * 50 + 60], 1)
    boxes = np.concatenate([rng.random((20, 2)) * 40, rng.random((20, 2)) * 40 + 55], 1)
    assert np.allclose(decode(encode(boxes, anchors), anchors), boxes)
    assert np.allclose(encode(anchors, anchors), 0)
