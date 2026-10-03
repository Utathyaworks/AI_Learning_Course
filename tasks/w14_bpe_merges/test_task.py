import numpy as np
from submission import *  # noqa: F401,F403
def test_classic_example():
    counts = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    m = learn_bpe(counts, 4)
    assert m[0] == ("e", "s")
    assert m[1] == ("es", "t")
    assert m[2] == ("est", "</w>")
    assert m[3] == ("l", "o")

def test_tie_break():
    assert learn_bpe({"ab": 1, "cd": 1}, 1) == [("a", "b")]

def test_stops_early():
    assert len(learn_bpe({"a": 3}, 10)) == 1
