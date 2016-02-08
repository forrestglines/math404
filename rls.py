import numpy as np
import scipy.linalg as la
def rls(A,B):
    x = np.ones(A.shape[1]) 
    K = np.ones((A.shape[1],A.shape[1]))
    for i in range(B.size):
        #import pdb;pdb.set_trace()
        a = A[:i+1]
        R = 1
        K = K - K.dot(a.T).dot(la.inv(R + a.dot(K).dot(a.T))).dot(a).dot(K)
        x = x - (K.dot(a.T)/R).dot(a.dot(x) - B[i])
    return x


def compare(A,B):
    
    xRls = rls(A,B)
    diffRls = la.norm(A.dot(xRls) - B)

    xOls = la.lstsq(A,B)[0]
    diffOls = la.norm(A.dot(xOls) - B)

    return (diffRls,diffOls)
