import numpy as np
from scipy.linalg import cholesky
from ..utils.svd import svd
from ..utils.svd import cutonRatio
from scipy.sparse.linalg import eigs

class LDA:
    def __init__(self, regu=False, reguAlpha = 0.1, reguType='ridge',
                 regularizerR=None, fisherFace=False, pcaRatio=1):
        # Linear Discriminanat Analysis
        # Input:
        #       regu:           是否使用正则化
        #       reguALpha:      正则化参数
        #       regeType:       正则化类型:
        #                       'ridge':岭回归
        #                       'custom':用户提供正则化矩阵
        #       regularizerR:   用户提供的矩阵 (nFeat * nFeat)
        #       fisherFace:     True: pcaRatio = nSamp - nFeat
        #                       Default: False
        #       pcaRatio:       The percentage of principal The percentage
        #                       of principal step. The percentage is
        #                       calculated based on the eigenvalue. Default is 1
        self.regu = regu
        self.reguAlpha = reguAlpha
        self.reguType = reguType
        self.regularizerR = regularizerR
        self.fisherFace = fisherFace
        self.pcaRatio = pcaRatio

    def fit(self, X, y):
        # LDA fit
        # Input:
        #       X:          nSamp * nFeat
        #       y:          labels
        # Output:
        #       eigVal:     eigvalues
        #       eigVec:     eigvectors
        (nSamp, nFeat) = X.shape
        labels = np.unique(y).flatten()
        nLabel = len(labels)
        n_neighbors = nLabel - 1

        if self.fisherFace:
            self.pcaRatio = nSamp - nLabel

        X = X - np.mean(X, axis=0)
        isPos = True
        try:
            DPrime = X.T.dot(X)
            DPrime = np.maximum(DPrime, DPrime.T)
            R = cholesky(DPrime)
            self.regu = False
        except Exception:
            isPos = False

        if not self.regu:
            (U, S, V) = svd(X)
            (U, S, V) = cutonRatio(U, S, V, self.pcaRatio)

            X = U
            eigVecPCA = V.dot(np.diag(1 / np.diag(S)))
        else:
            if not isPos:
                DPrime = X.T.dot(X)

                if self.reguType.lower() == 'ridge':
                    DPrime = DPrime + (np.eye(DPrime.shape[0]) * self.reguAlpha)
                elif self.reguType.lower() == 'custom':
                    DPrime = DPrime + (self.regularizerR * self.reguAlpha)
                else:
                    DPrime = DPrime + (self.regularizerR * self.reguAlpha)

                DPrime = np.maximum(DPrime, DPrime.T)

        (nSamp, nFeat) = X.shape
        Hb = np.zeros((nLabel, nFeat))
        for i in range(nLabel):
            index = (y == labels[i]).flatten()
            mean = np.mean(X[index, :], axis=0)
            Hb[i, :] = np.sqrt(len(index)) * mean

        if not self.regu:
            (U, S, V) = svd(Hb)
            eigVal = np.diag(S)
            eigIdx = (eigVal >= 1e-3)
            eigVec = V[:, eigIdx]
            eigVal = eigVal[eigIdx]

            eigVal = np.sqrt(eigVal)
            eigVec = eigVecPCA.dot(eigVec)
        else:
            WPrime = Hb.T.dot(Hb)
            WPrime = np.maximum(WPrime, WPrime.T)

            dimMatrix = WPrime.shape[1]
            if n_neighbors > dimMatrix:
                n_neighbors = dimMatrix

            if isPos:
                eigVal, eigVec = eigs(A=WPrime, M=R, k=n_neighbors, which='LR')
            else:
                eigVal, eigVec = eigs(A=WPrime, M=DPrime, k=n_neighbors, which='LR')

        eigVec = eigVec / np.linalg.norm(eigVec, axis=0)

        self.eigVec = eigVec
        self.eigVal = eigVal

    def transform(self, X):
        return X.dot(self.eigVec)