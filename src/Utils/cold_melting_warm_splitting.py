import numpy as np


def ratings_per_user(URM):
    a = np.asmatrix(np.ones(shape=URM.shape[1])).T

    return URM.dot(a)


def cold_melting_warm_splitting(URM, threshold=6.0):
    rpu = ratings_per_user(URM)

    return np.nonzero(rpu == 0), np.nonzero(np.logical_or(0 < rpu, rpu <= threshold)), np.nonzero(rpu > threshold)