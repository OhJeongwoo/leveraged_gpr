import numpy as np
import random
import sklearn.metrics.pairwise
import scipy.spatial.distance
import pickle

with open("../data/expert_lev.pkl", 'rb') as pk:
        data = pickle.load(pk)

print(data)