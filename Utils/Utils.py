
import numpy as np
import scipy as sc
import os


def PermuteMatrix(matrix, reference):
    """
    Permute rows and columns of 'matrix' according to the order given in 
    'reference'.
    """
    mh = np.zeros((len(matrix), 1))
    sh = set()
    for i in range(len(matrix)):
        if(not reference[i] in sh):
            mh = np.concatenate((mh, matrix[:,i][:, np.newaxis]), 1)
            sh.add(reference[i])
            for j in range(i+1, len(matrix)):
                if(reference[j] == reference[i]):                    
                    mh = np.concatenate((mh, matrix[:,j][:, np.newaxis]), 1)
                    
    mh = mh[:,1:]
    
    mv = np.zeros((1, len(mh)))
    sh = set()
    for i in range(len(mh)):
        if(not reference[i] in sh):
            mv = np.concatenate((mv, mh[i,:][np.newaxis, :]), 0)
            sh.add(reference[i])
            for j in range(i+1, len(mh)):
                if(reference[j] == reference[i]):                    
                    mv = np.concatenate((mv, mh[j,:][np.newaxis, :]), 0)
                    
    mv = mv[1:,:]
    return mv  