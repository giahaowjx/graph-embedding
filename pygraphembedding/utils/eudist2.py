import numpy as np

def eudist2(featA, featB=None, bSqrt=True):
    # 计算数据的距离矩阵
    # Input:
    #       feat_a:     nSample_a * nFeature
    #       feat_b:     nSample_b * nFeature
    #       bSqrt:      是否开方
    # Return:
    #       D:          nSample_a * nSample_b
    #              or   nSample_a * nSample_b
    if featB == None:
        aa = np.sum(featA * featA, axis=1)
        ab = np.dot(featA, featA.T)

        D = np.add(aa, aa.T) - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)

        D = np.maximum(D, D.T)

    else:
        aa = np.sum(featA, featA, axis=1)
        bb = np.sum(featB, featB, axis=1)
        ab = np.dot(featA, featB.T)

        D = np.add(aa, bb.T) - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)

    return D