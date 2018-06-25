import unittest
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from knnFeat import knnExtract


class TestKnnFeat(unittest.TestCase):
    def test_knnFeat(self):
        X = np.reshape(np.array([0, 1, 3, 4, 5, 6, 1, 1, 0, 3]), (5, 2))
        y = np.array([0, 0, 0, 1, 1])

        expected = knnExtract(X, y, k = 1, holds = 5)

        self.assertEqual(expected.shape, X.shape)

if __name__ == "__main__":
    unittest.main()