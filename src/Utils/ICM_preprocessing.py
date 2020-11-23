from scipy import sparse as sps
import numpy as np

def combine(ICM: sps.csr_matrix, URM : sps.csr_matrix):
    return sps.hstack((URM.T, ICM), format='csr')

def binarize(x):
    if x != 0:
        return 1
    return x

def binarize_ICM(ICM: sps.csr_matrix):
    vbinarize = np.vectorize(binarize)

    ICM.data = vbinarize(ICM.data)