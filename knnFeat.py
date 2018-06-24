import numpy as np
from sklearn.model_selection import KFold

def _distance(a, b):
    return np.linalg.norm(b - a)

def _get_feat(data, X_train, y_train, class_index, k_index):
    inclass_X = X_train[y_train == class_index]
    distances = np.array([_distance(a, data) for a in inclass_X])
    sorted_distances_index = np.argsort(distances)
    nearest_index = list(sorted_distances_index[0: (k_index + 1)])
    dist = np.sum(distances[nearest_index])
    return dist

def knnExtract(X, y, k = 1, holds = 5):
    CLASS_NUM = len(set(y))
    res = np.empty((len(X), CLASS_NUM * k))
    kf = KFold(n_splits = holds,  shuffle = True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        features = np.empty([0, len(X_test)])
        
        for class_index in range(CLASS_NUM):
            for k_index in range(k):
                feat = np.array([np.apply_along_axis(_get_feat, 1, X_test, X_train, y_train, class_index, k_index)])
                features = np.append(features, feat, axis = 0)
        res[test_index] = features.T            

    return res
