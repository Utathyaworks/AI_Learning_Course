import numpy as np
from submission import *  # noqa: F401,F403
def test_basic():
    assert wer("the cat sat", "the cat sat") == 0.0
    assert abs(wer("the cat sat on the mat", "the cat sit on mat") - 2 / 6) < 1e-12

def test_insertions_can_exceed_one():
    assert wer("hi", "oh hi there friend") == 3.0

def test_empty_reference():
    assert wer("", "") == 0.0 and wer("", "x") == float("inf")

def test_whitespace():
    assert wer("a  b\tc", "a b c") == 0.0
