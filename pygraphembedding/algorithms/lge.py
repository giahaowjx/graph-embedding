import numpy as np
from scipy.linalg import cholesky
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from ..utils.svd import svd
from ..utils.svd import cutonRatio

def lge(data, W, D=None, reduceDim=30, regu=False, reguAlpha=0.1, reguType='Ridge',
        regularizerR=None, pcaRatio=1):
    # Linear Graph Embedding
    #
    # Input:
    #       data:       data matrix. Each row vector of data is a sample vector
    #       W:          Affinity graph matrix.
    #       D:          Constraint graph matrix.
    #
    #       LGE solves the optimization problem of
    #       a* = argmax (a'data'WXa)/(a'data'DXa)
    #       Default: D = I
    #
    # Optional input:
    #       reduceDim:  The dimensionality of the reduced subspace. If 0, all the
    #                   dimensions will be kept. Default is 30.
    #       regu:       True: regularized solution
    #                       a* = argmax (a'X'WXa)/(a'X'DXa+ReguAlpha*I)
    #                   False： solve the sinularity problem by SVD, default: False
    #       reguAlpha:  The regularization parameter. Valid when Regu==1. Default
    #                   value is 0.1.
    #       reguType:   'Ridge': Tikhonov regularization
    #                   'Custom': User provided regularization matrix
    #                   Default: 'Ridge'
    #       regularizerR:(nFea x nFea) regularization matrix which should be provided
    #                    if ReguType is 'Custom'. nFea is the feature number of data
    #                    matrix
    #       pcaRatio:   The percentage of principal component kept in the PCA
    #                   step. The percentage is calculated based on the eigenvalue.
    #                   Default is 1
    #                   (100%, all the non-zero eigenvalues will be kept. If PCARatio > 1
    #                   , the PCA step will keep exactly PCARatio principle components
    #                   (does not exceed the exact number of non-zero components).
    # Output:
    #       eigVectors: Each column is an embedding function, for a new sample vector
    #                   (row vector) x,  y = x*eigvector will be the embedding result of x.
    #       eigValue:   The sorted eigvalue of the eigen-problem.

    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1

    print("[INFO] Inside the LGE function...")

    # 判断传入参数是否合理
    (nSamp, nFeat) = data.shape
    if W.shape[0] != nSamp:
        print("[ERR] W and data mismatch!")
        exit(1)
    if D is None and D.shape[0] != nSamp:
        print("[ERR] D and data mismatch!")
        exit(1)

    isPos = True

    if not regu and nSamp > nFeat and pcaRatio >= 1:
        if D is not None:
            DPrime = data.T.dot(D).dot(data)
        else:
            DPrime = data.T.dot(data)
        DPrime = np.maximum(DPrime, DPrime.T)

        try:
            R = cholesky(DPrime)
            # 这里将regu置1是为了不对D进行奇异值分解，且会通过isPos跳过加入正则矩阵的步骤
            regu = True
        except Exception:
            isPos = False
            print("[ERR] Cholesky Decomposition Failed!")

    # SVD
    if not regu:
        (U, S, V) = svd(data)
        (U, S, V) = cutonRatio(U, S, V, pcaRatio)
        eigValPCA = np.diag(S)
        if D is not None:
            data = U.dot(S)
            eigVecPCA = V

            DPrime = data.T.dot(D).dot(data)
            DPrime = np.maximum(DPrime, DPrime.T)
        else:
            data = U
            eigVecPCA = V.dot(np.diag(1 / eigValPCA))
    else:
        if not isPos:
            if D != None:
                DPrime = data.T.dot(D).dot(data)
            else:
                DPrime = data.T.dot(data)

            if reguType.lower() == 'ridge':
                if reguAlpha > 0:
                    for i in np.arange(DPrime.shape[0]):
                        DPrime[i, i] = DPrime[i, i] + reguAlpha
            elif reguType.lower() == 'tensor':
                if reguAlpha > 0:
                    DPrime = DPrime + reguAlpha * regularizerR
            elif reguType.lower() == 'custom':
                if reguAlpha > 0:
                    DPrime = DPrime + reguAlpha * regularizerR
            else:
                print("[ERR] reguType does not exist!")

    WPrime = data.T.dot(W).dot(data)
    WPrime = np.maximum(WPrime, WPrime.T)

    # Generalized Eigen
    dimMatrix = WPrime.shape[1]

    if reduceDim > dimMatrix:
        reduceDim = dimMatrix

    if dimMatrix > MAX_MATRIX_SIZE and reduceDim < dimMatrix * EIGVECTOR_RATIO:
        bEigs = True
    else:
        bEigs = False

    if bEigs:
        if not regu and D == None:
            eigVal, eigVec = eigs(A=WPrime, k=reduceDim, which='LR')
        else:
            if isPos:
                eigVal, eigVec = eigs(A=WPrime, M=R, k=reduceDim, which='LR')
            else:
                eigVal, eigVec = eigs(A=WPrime, M=DPrime, k=reduceDim, which='LR')
        eigVal = np.diag(eigVal)

    else:
        if not regu and D is None:
            eigVal, eigVec = eig(a=WPrime)
        else:
            eigVal, eigVec = eig(a=WPrime, b=DPrime)

        index = np.argsort(-eigVal)
        eigVal = eigVal[index]
        eigVec = eigVec[:, index]

        if reduceDim < eigVec.shape[1]:
            eigVec = eigVec[:, :reduceDim]

    if not regu:
        eigVec = eigVecPCA.dot(eigVec)

    eigVec = eigVec / np.linalg.norm(eigVec, axis=0)

    return eigVal, eigVec