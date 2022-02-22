import numpy as np
import math

def normalize(X):
    """
    input
    X: before normalize (N x K)

    output
    X: after normalize (N x K)
    """
    mu = np.mean(X, axis=0)
    sig = np.std(X, axis=0)
    return (X-mu) / sig