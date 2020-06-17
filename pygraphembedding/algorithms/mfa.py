from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigs
from scipy.sparse import csc_matrix
from pygraphembedding.utils import eudist2
from .lge import lge
import numpy as np


class MFA:
    def __init__(self, intraK, interK, n_neighbors):
        self.intraK = intraK
        self.interK = interK
        self.n_neighbors = n_neighbors

    def fit(self, X, y, keepMean=False, regu=False):
        # Marginal Fisher Analysis
        # Input:
        #       X:              Data matrix. Each row vector of fea is a data point
        #       y:              label vector
        # Optional parameters:
        #       intraK:         0:
        #                           Sc: Put an edge between two nodes if and only if
        #                               and only if they belong to same class
        #                      >0:
        #                           Sc: Put an edge between two nodes if they belong
        #                               to same class and they are among the intraK
        #                               nearst neighbors of each other in this class
        #                       Default: 5
        #       interK:         0:
        #                           Sp: Put an edge between two nodes if and only if
        #                               they belong to different classes.
        #                       >0:
        #                           Sp: Put an edge between two nodes if they rank
        #                               top interK pairs of all the distance pair
        #                               of samples belong to different classes
        #                       Default: 20
        #       keepMean:       Keep data  mean? Default: False
        # Output:
        #       eigVal:         The eigVal of LPP eigen-problem, sorted from smallest
        #                       to largest
        #       eigVec:         Each column is an embedding function, for a new data
        #                       point x, y = x.dot(eigVec) will be the embedding result
        (nSamp, nFeat) = X.shape
        if len(y) != nSamp:
            print('[ERR] labels and data mismathch')
            exit(1)

        labels = np.unique(y)
        nLabel = len(labels)

        D = eudist2(X, bSqrt=False)

        nIntraPair = 0
        if self.intraK > 0:
            G = np.zeros((nSamp * (self.intraK + 1), 3))
            idNow = 0
            for i in np.arange(nLabel):
                classIdx = np.argwhere(y == labels[i]).flatten()
                DClass = D[classIdx]
                DClass = DClass[:, classIdx]

                index = np.argsort(DClass, axis=1)
                nClassNow = len(classIdx)
                nIntraPair = nIntraPair + nClassNow ** 2
                if self.intraK < nClassNow:
                    index = index[:, :self.intraK + 1]
                else:
                    index = np.c_[index, np.repeat(index[:,-1].reshape((nClassNow, 1)),
                                                   self.intraK + 1 - nClassNow, axis=1)]

                nSampClass = DClass.shape[0] * (self.intraK + 1)
                G[idNow:nSampClass+idNow, 0] = np.repeat(classIdx, self.intraK + 1)
                G[idNow:nSampClass+idNow, 1] = classIdx[index].flatten()
                G[idNow:nSampClass+idNow, 2] = 1
                idNow = idNow + nSampClass

            Sc = csc_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSamp, nSamp))
            Sc = Sc.toarray()
            Sc = np.maximum(Sc, Sc.T)
        else:
            Sc = np.zeros(nSamp, nSamp)
            for i in np.arange(nLabel):
                classIdx = (y == labels[i])
                nClassNow = len(classIdx)
                nIntraPair = nIntraPair + nClassNow ** 2
                subSc = Sc[classIdx]
                subSc[:, classIdx] = 1
                Sc[classIdx] = subSc

        if self.interK > 0 and self.interK < (nSamp ** 2 - nIntraPair):
            maxD = np.max(D) + 100
            for i in np.arange(nLabel):
                classIdx = np.argwhere(y==labels[i]).flatten()
                subD = D[classIdx]
                subD[:, classIdx] = maxD
                D[classIdx] = subD

            idx = np.argsort(D, axis=1)
            idx = idx[:, :self.interK].T
            Sp = np.zeros((nSamp, nSamp))
            Sp[np.arange(nSamp), idx] = 1
            Sp = np.maximum(Sp, Sp.T)
        else:
            Sp = np.ones((nSamp, nSamp))
            for i in np.arange(nLabel):
                classIdx = (y == labels[i])
                subSp = Sp[classIdx]
                subSp[:, classIdx] = 0
                Sp[classIdx] = subSp

        Sp = np.diag(np.sum(Sp, axis=1)) - Sp
        Sc = np.diag(np.sum(Sc, axis=1)) - Sc

        if not keepMean:
            X = X - np.mean(X, axis=0)

        eigVal, eigVec = lge(X, Sp, Sc, reduceDim=self.n_neighbors, regu=regu)

        self.eigVal = eigVal
        self.eigVec = eigVec

    def transform(self, X):
        return np.dot(X, self.eigVec)