import numpy as np
from submission import *  # noqa: F401,F403
def _W(seed=0, V=11, d=16, dh=8, n=12):
    r = np.random.default_rng(seed)
    return dict(E=r.normal(size=(V, d)), P=0.1 * r.normal(size=(n, d)), WQ=r.normal(size=(d, dh)) / 4,
                WK=r.normal(size=(d, dh)) / 4, WV=r.normal(size=(d, dh)) / 4, WO=r.normal(size=(dh, d)) / 4,
                U=r.normal(size=(d, V)))

def test_forward_shapes_and_causality():
    W = _W(); t = np.array([1, 4, 2, 7, 3])
    lg, c = forward(t, W)
    assert lg.shape == (5, 11) and np.allclose(c["A"].sum(1), 1) and np.allclose(np.triu(c["A"], 1), 0)
    lg2, _ = forward(np.array([1, 4, 2, 7, 9]), W)
    assert np.allclose(lg[:4], lg2[:4])          # changing the last token can't change earlier positions

def test_circuits():
    W = _W()
    assert qk_circuit(W).shape == (11, 11) and ov_circuit(W).shape == (11, 11)
    assert np.allclose(ov_circuit(W)[3], W["E"][3] @ W["WV"] @ W["WO"] @ W["U"])

def test_dla_sums_to_logit():
    W = _W(); t = np.array([5, 1, 8, 2, 2, 6])
    lg, c = forward(t, W)
    e, h = direct_logit_attribution(c, W, 4, 7)
    assert np.isclose(e + h, lg[4, 7])

def test_patching():
    W = _W(1); clean = np.array([3, 5, 1, 9, 2]); corrupt = np.array([3, 5, 1, 9, 2])
    corrupt = corrupt.copy(); corrupt[1] = 7
    assert np.isclose(patching_effect(clean, corrupt, W, 4, 0), 1.0)   # the head is the only path that sees token 1 at pos 4
    lg, c = forward(clean, W)
    same, _ = forward(clean, W, patch={"head": c["head"]})
    assert np.allclose(same, lg)
