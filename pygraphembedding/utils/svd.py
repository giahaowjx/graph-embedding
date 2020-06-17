import numpy as np
from scipy.linalg import eig
from scipy.sparse.linalg import eigs

def svd(X, reducedDim=0):
    # Accelerated singular value decomposition.
    #
    # [U,S,V] = mySVD(X) produces a diagonal matrix S, of the
    # dimension as the rank of X and with nonnegative diagonal elements in
    # decreasing order, and unitary matrices U and V so that
    # X = U*S*V.T
    #
    # Input:
    #       X:              nSamp * nFeat
    #       reducedDime:    Default: 0
    # Return:
    #       U:              Left Singular Matrix
    #       V:              Right Singular Matrix
    #       S:              Dialog Matrix with eignvalue

    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1

    (nSamp, nFeat) = X.shape
    if nFeat / nSamp > 1.0713:
        data = X.dot(X.T)
        data = np.maximum(data, data.T)

        dimMatrix = data.shape[0]
        if reducedDim > 0 and dimMatrix > MAX_MATRIX_SIZE and \
            reducedDim < dimMatrix * EIGVECTOR_RATIO:
            (eigVal, eigVec) = eigs(A = data, k=reducedDim, which='LR')
        else:
            (eigVal, eigVec) = eig(data)
            # 取对应最大特征值的特征向量
            index = np.argsort(-eigVal)
            eigVal = eigVal[index]
            eigVec = eigVec[:, index]

        maxEigVal = np.max(np.abs(eigVal))
        eigIdx = (np.abs(eigVal) / maxEigVal) >= 1e-10
        eigVal = eigVal[eigIdx]
        U = eigVec[:, eigIdx]

        if reducedDim > 0 and reducedDim < len(eigVal):
            eigVal = eigVal[:reducedDim]

        eigValHalf = np.sqrt(eigVal)
        S = np.diag(eigValHalf)

        eigValMinusHalf = 1 / eigValHalf
        V = X.T.dot(np.multiply(U, eigValMinusHalf))

        return U, S, V
    else:
        data = X.T.dot(X)
        data = np.maximum(data, data.T)

        dimMatrix = data.shape[0]
        if reducedDim > 0 and dimMatrix > MAX_MATRIX_SIZE and \
                reducedDim < dimMatrix * EIGVECTOR_RATIO:
            (eigVal, eigVec) = eigs(A=data, k=reducedDim, which='LR')

        else:
            (eigVal, eigVec) = eig(data)
            index = np.argsort(-eigVal)
            eigVal = eigVal[index]
            eigVec = eigVec[:, index]

        maxEigValue = np.max(np.abs(eigVal))
        eigIdx = (np.abs(eigVal) / maxEigValue) >= 1e-10
        eigVal = eigVal[eigIdx]
        V = eigVec[:, eigIdx]

        if reducedDim > 0 and reducedDim < len(eigVal):
            eigVal = eigVal[:reducedDim]
            V = eigVec[:,:reducedDim]

        eigValHalf = np.sqrt(eigVal)
        S = np.diag(eigValHalf)

        eigValMinusHalf = 1 / eigValHalf
        U = X.dot(np.multiply(V, eigValMinusHalf))

        return U, S, V

def cutonRatio(U, S, V, pcaRatio=1):
    eigValPCA = np.diag(S)
    if pcaRatio > 1 and pcaRatio < eigValPCA.shape[0]:
        U = U[:, :pcaRatio]
        V = V[:, :pcaRatio]
        S = S[:pcaRatio, :pcaRatio]

    elif pcaRatio < 1:
        sumEig = np.sum(eigValPCA)
        sumEig = sumEig * pcaRatio
        sumNow = 0
        for idx in np.arange(len(eigValPCA)):
            sumNow = sumNow + eigValPCA[idx]
            if sumNow >= sumEig:
                U = U[:, :idx]
                V = V[:, :idx]
                S = S[:idx, :idx]
                break

    return U, S, V