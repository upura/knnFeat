import numpy as np
import sys
import os
from knnFeat import _get_feat
sys.path.append(os.getcwd())


# Case 1: class_index == 0 and k_index == 0
def test_get_feat_c0k0():
    data = np.array([0, 0])
    X_train = np.reshape(np.array([0, 1, 3, 4, 5, 6, 1, 1, 0, 3]), (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])
    class_index = 0
    k_index = 0

    expected = _get_feat(data, X_train, y_train, class_index, k_index)

    # [0, 1] is the 1-nearest point
    actual = 1

    assert expected == actual


# Case 2: class_index == 0 and k_index == 1
def test_get_feat_c0k1():
    data = np.array([0, 0])
    X_train = np.reshape(np.array([0, 1, 3, 4, 5, 6, 1, 1, 0, 3]), (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])
    class_index = 0
    k_index = 1

    expected = _get_feat(data, X_train, y_train, class_index, k_index)

    # [0, 1] and [3, 4] is the 2-nearest points
    actual = 1 + 5

    assert expected == actual


# Case 3: class_index == 1 and k_index == 0
def test_get_feat_c1k0():
    data = np.array([0, 0])
    X_train = np.reshape(np.array([0, 1, 3, 4, 5, 6, 0, 2, 0, 3]), (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])
    class_index = 1
    k_index = 0

    expected = _get_feat(data, X_train, y_train, class_index, k_index)

    # [0, 2] is the 1-nearest point
    actual = 2

    assert expected == actual


# Case 4: class_index == 1 and k_index == 1
def test_get_feat_c1k1():
    data = np.array([0, 0])
    X_train = np.reshape(np.array([0, 1, 3, 4, 5, 6, 0, 2, 0, 3]), (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])
    class_index = 1
    k_index = 1

    expected = _get_feat(data, X_train, y_train, class_index, k_index)

    # [0, 2] and [0, 3]is the 2-nearest points
    actual = 2 + 3

    assert expected == actual
