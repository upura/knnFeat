import numpy as np
import sys
import os
from knnFeat import _distance
sys.path.append(os.getcwd())


def test_distance():
    a = np.array([0, 0])
    b = np.array([3, 4])
    expected = _distance(a, b)
    actual = 5
    assert expected == actual
