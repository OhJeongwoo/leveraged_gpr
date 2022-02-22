import numpy as np
import math
import sklearn.metrics.pairwise

class Kernel:
    def __init__(self, g=None, l=None, w=None):
        if g is not None:
            self.g = g
        else:
            self.g = 1.0
        if l is not None:
            self.l = l
        else:
            self.l = 1.0
        if w is not None:
            self.w = w
        else:
            self.w = 0.01
        
    
    def psd_kernel(self, x1, x2, delta):
        """
        input
        x1, x2: input variable (|S|, )
        delta: delta(i,j), True, iff i == j
        
        output
        rbf kernel fuction value
        """
        if x1.shape[0] != x2.shape[0]:
            print("[ERR] mismatch dimension. x1: %d, x2: %d" %(x1.shape[0], x2.shape[0]))
        nm = np.linalg.norm(x1-x2)
        return self.g ** 2 * math.exp(-0.5 * nm ** 2 / self.l ** 2) + self.w ** 2 * delta

    def sl_kernel(self, r1, r2, x1, x2, delta):
        """
        input
        r1, r2: leveraged value (scalar)
        x1, x2: input variable (|S|,)
        delta: delta(i,j), True, iff i == j
        
        output
        smooth leveraged kernel function value
        """
        if x1.shape[0] != x2.shape[0]:
            print("[ERR] mismatch dimension. x1: %d, x2: %d" %(x1.shape[0], x2.shape[0]))
        lev_coeff = math.cos(0.5 * math.pi * (r1 - r2))
        psd_kernel_value = self.psd_kernel(x1, x2, delta)
        return lev_coeff * psd_kernel_value

    def kernel_matrix(self, lev, X):
        """
        input
        lev: leveraged value (N x 1)
        X: input dataset (N x |S|)
        
        output
        K: kernel matrix (N x N)
        """
        if lev.shape[0] != X.shape[0]:
            print("[ERR] mismatch dimension. lev: %d, X: %d" %(lev.shape[0], X.shape[0]))
        N = X.shape[0]
        D = sklearn.metrics.pairwise_distances(X, X)
        gamma = sklearn.metrics.pairwise_distances(lev.reshape(N,1), lev.reshape(N,1))
        K = np.cos(0.5 * np.pi * gamma) * (self.g ** 2 * np.exp(-0.5 * D ** 2 / self.l ** 2) + self.w ** 2 * np.eye(N))
        # for i in range(N):
        #     for j in range(N):
        #         if i == j:
        #             delta = True
        #         else:
        #             delta = False
        #         K[i][j] = self.sl_kernel(lev[i], lev[j], X[i], X[j], delta)
        return K
