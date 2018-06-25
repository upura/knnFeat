import unittest
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from knnFeat import _distance


class TestDistance(unittest.TestCase):
    def test_distance(self):
        a = np.array([0, 0])
        b = np.array([3, 4])
        expected = _distance(a, b)
        actual = 5
        self.assertEqual(expected, actual)

if __name__ == "__main__":
    unittest.main()