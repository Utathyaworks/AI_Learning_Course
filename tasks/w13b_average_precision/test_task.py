import numpy as np
from submission import *  # noqa: F401,F403
A = [0, 0, 10, 10]; B = [20, 20, 30, 30]; FAR = [50, 50, 60, 60]

def test_perfect():
    gt = {0: np.array([A]), 1: np.array([B])}
    assert average_precision([(0, 0.9, A), (1, 0.8, B)], gt) == 1.0

def test_hand_example():
    gt = {0: np.array([A]), 1: np.array([B])}
    dets = [(0, 0.9, A), (1, 0.8, FAR), (1, 0.7, B)]
    assert np.isclose(average_precision(dets, gt), 0.5 * 1.0 + 0.5 * (2 / 3))

def test_duplicates_are_false_positives():
    gt = {0: np.array([A])}
    dets = [(0, 0.9, A), (0, 0.8, [0, 0, 10, 9.5])]
    assert np.isclose(average_precision(dets, gt), 1.0)
    assert np.isclose(average_precision([(0, 0.9, [0, 0, 10, 9.5]), (0, 0.95, FAR)], gt), 0.5)

def test_iou_threshold():
    gt = {0: np.array([A])}
    shifted = [4, 0, 14, 10]   # IoU = 6/14
    assert average_precision([(0, 0.9, shifted)], gt, 0.5) == 0.0
    assert average_precision([(0, 0.9, shifted)], gt, 0.4) == 1.0

def test_no_ground_truth():
    assert average_precision([(0, 0.5, A)], {}) == 0.0
