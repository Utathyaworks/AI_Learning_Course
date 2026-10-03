import numpy as np
from submission import *  # noqa: F401,F403
def test_mask_and_labels():
    pw = np.array([[0.97, 0.03], [0.6, 0.4], [0.01, 0.99]])
    loss, mask, pl = fixmatch_unlabeled_loss(pw, np.zeros((3, 2)), 0.95)
    assert list(mask) == [True, False, True] and list(pl) == [0, 0, 1]
    assert abs(loss - 2 * np.log(2) / 3) < 1e-12

def test_no_confident_examples():
    loss, mask, _ = fixmatch_unlabeled_loss(np.full((4, 3), 1 / 3), np.random.default_rng(0).normal(size=(4, 3)))
    assert loss == 0.0 and not mask.any()

def test_stable():
    pw = np.array([[1.0, 0.0]]); zs = np.array([[-1e4, 1e4]])
    loss, _, _ = fixmatch_unlabeled_loss(pw, zs)
    assert np.isfinite(loss) and abs(loss - 2e4) < 1e-6
