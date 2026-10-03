import numpy as np
from submission import *  # noqa: F401,F403
def test_init_is_identity():
    rng = np.random.default_rng(0)
    W = rng.normal(size=(8, 6)); x = rng.normal(size=(3, 8))
    A, B = lora_init(8, 6, 2)
    assert A.shape == (8, 2) and B.shape == (2, 6) and np.all(B == 0)
    assert np.allclose(lora_forward(x, W, A, B, 16), x @ W)

def test_merge_matches_forward():
    rng = np.random.default_rng(1)
    W, A, B, x = rng.normal(size=(8, 6)), rng.normal(size=(8, 4)), rng.normal(size=(4, 6)), rng.normal(size=(5, 8))
    assert np.allclose(x @ lora_merge(W, A, B, 8), lora_forward(x, W, A, B, 8))

def test_update_rank_and_count():
    rng = np.random.default_rng(2)
    A, B = rng.normal(size=(64, 3)), rng.normal(size=(3, 32))
    delta = lora_merge(np.zeros((64, 32)), A, B, 3)
    assert np.linalg.matrix_rank(delta) == 3
    assert lora_param_count(4096, 4096, 16) == 131072
