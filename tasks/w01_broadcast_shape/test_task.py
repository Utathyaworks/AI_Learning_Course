import numpy as np
from submission import *  # noqa: F401,F403
import pytest

def test_classic():
    assert broadcast_shape((8, 1, 6, 1), (7, 1, 5)) == (8, 7, 6, 5)

def test_scalar_and_many():
    assert broadcast_shape((), (3, 1), (1, 4)) == (3, 4)

def test_random_agrees_with_numpy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = rng.integers(1, 4)
        shapes = [tuple(int(x) for x in rng.choice([1, 2, 3], size=rng.integers(0, 4))) for _ in range(k)]
        try:
            expected = np.broadcast_shapes(*shapes)
        except ValueError:
            with pytest.raises(ValueError):
                broadcast_shape(*shapes)
        else:
            assert broadcast_shape(*shapes) == expected

def test_incompatible():
    with pytest.raises(ValueError):
        broadcast_shape((3,), (4,))

def test_zero_size():
    assert broadcast_shape((0, 1), (1, 5)) == (0, 5)
