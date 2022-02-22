from kernel import Kernel

import numpy as np
import math
import time

EPSILON = 1e-6

class LGPR:
    def __init__(self, X, y):
        """
        input
        X: input variable (N * |S|)
        y: output variable (N * |A|)
        """
        if X.shape[0] != y.shape[0]:
            print("[ERR] mismatch dimension. X: %d, y: %d" %(X.shape[0], y.shape[0]))
        self.N = X.shape[0]
        self.A = y.shape[1]
        self.lev = np.ones(self.N)
        self.kernel = Kernel()
        self.X = X
        self.y = y
        self.n_sample = 1000
        self.step = 21
        self.candidate = np.linspace(-1.0, 1.0, self.step)
        self.log_step = 100
        self.cur_step = 0
        self.init_time = time.time()




    def init_kernel(self):
        d_list = []
        t = 0
        while t < self.n_sample:
            i = np.random.randint(self.N)
            j = np.random.randint(self.N)
            if i == j:
                continue
            t += 1
            d_list.append(np.linalg.norm(self.X[i]-self.X[j]))
        d_list = sorted(d_list)
        self.kernel.l = d_list[self.n_sample // 2]

    def calculate_loss(self):
        K = self.kernel.kernel_matrix(self.lev, self.X)
        rt = 0.0
        s = time.time()
        for i in range(self.A):
            rt += 0.5 * np.transpose(self.y[:,i]) @ np.linalg.inv(K) @ self.y[:,i]
        rt += 0.5 * math.log(np.linalg.det(K) + EPSILON)
        return rt

    def solve(self):
        for idx in range(self.N):
            min_idx = None
            min_val = None
            for i in range(self.step):
                self.lev[idx] = self.candidate[i]
                L = self.calculate_loss()
                if min_val is None or min_val > L:
                    min_idx = i
                    min_val = L
            self.lev[idx] = self.candidate[min_idx]

            if (idx + 1) / self.N * 100 + EPSILON > 100 / self.log_step * self.cur_step:
                print("[%.3f] LGPR %.2f %% completed" %(time.time() - self.init_time, 100 / self.log_step * self.cur_step))
                self.cur_step += 1
        return self.lev
