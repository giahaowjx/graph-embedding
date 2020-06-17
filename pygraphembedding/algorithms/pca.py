import numpy as np
from ..utils.svd import  svd

class PCA:
    def __init__(self, n_neighbors=0):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        # PCA
        # Input:
        #       X:          nSamp * nFeat
        # Output:
        #       eigVal:     EigVals
        #       eigVec:     EigVector
        (nSamp, nFeat) = X.shape
        if self.n_neighbors > nFeat or self.n_neighbors <= 0:
            self.n_neighbors = nFeat

        X = X - np.mean(X, axis=0)

        (eigVec, eigVal, _) = svd(X=X.T, reducedDim=self.n_neighbors)
        eigVal = np.square(np.diag(eigVal))

        self.eigVal = eigVal
        self.eigVec = eigVec

    def transform(self, X):
        return X.dot(self.eigVec)