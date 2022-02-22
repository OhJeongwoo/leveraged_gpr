import os
import sys

from kernel import Kernel
from lgpr import LGPR
from utils import normalize

import numpy as np
import math
import time
import pickle

PROJECT_PATH = os.path.abspath("..")
DATA_PATH = PROJECT_PATH + "/data/"

if __name__ == "__main__":
    init_time = time.time()

    N_EXPERT = 60
    N_UNKNOWN = 100

    expert_state_file_name = 'expert_s'
    expert_state_file = DATA_PATH + expert_state_file_name + ".pkl"
    with open(expert_state_file, 'rb') as pk:
        expert_state = pickle.load(pk)
    
    expert_action_file_name = 'expert_a'
    expert_action_file = DATA_PATH + expert_action_file_name + ".pkl"
    with open(expert_action_file, 'rb') as pk:
        expert_action = pickle.load(pk)

    
    unknown_state_file_name = 'expert_s'
    unknown_state_file = DATA_PATH + unknown_state_file_name + ".pkl"
    with open(unknown_state_file, 'rb') as pk:
        unknown_state = pickle.load(pk)
    
    unknown_action_file_name = 'expert_a'
    unknown_action_file = DATA_PATH + unknown_action_file_name + ".pkl"
    with open(unknown_action_file, 'rb') as pk:
        unknown_action = pickle.load(pk)

    # unknown_state_file_name = 'fail_s'
    # unknown_state_file = DATA_PATH + unknown_state_file_name + ".pkl"
    # with open(unknown_state_file, 'rb') as pk:
    #     unknown_state = pickle.load(pk)
    
    # unknown_action_file_name = 'fail_a'
    # unknown_action_file = DATA_PATH + unknown_action_file_name + ".pkl"
    # with open(unknown_action_file, 'rb') as pk:
    #     unknown_action = pickle.load(pk)

    # unknown_state_file_name = 'expert_s'
    # unknown_state_file = DATA_PATH + unknown_state_file_name + ".pkl"
    # with open(unknown_state_file, 'rb') as pk:
    #     unknown_state = pickle.load(pk)
    
    # unknown_action_file_name = 'expert_a'
    # unknown_action_file = DATA_PATH + unknown_action_file_name + ".pkl"
    # with open(unknown_action_file, 'rb') as pk:
    #     unknown_action = pickle.load(pk)

    # unknown_state_file_name = 'expert_s'
    # unknown_state_file = DATA_PATH + unknown_state_file_name + ".pkl"
    # with open(unknown_state_file, 'rb') as pk:
    #     unknown_state = pickle.load(pk)
    
    # unknown_action_file_name = 'expert_a'
    # unknown_action_file = DATA_PATH + unknown_action_file_name + ".pkl"
    # with open(unknown_action_file, 'rb') as pk:
    #     unknown_action = pickle.load(pk)

    expert_state = np.array(expert_state[0:N_EXPERT])
    expert_action = np.array(expert_action[0:N_EXPERT])
    unknown_state = np.array(unknown_state[0:N_UNKNOWN])
    unknown_action = np.array(unknown_action[0:N_UNKNOWN])

    S = expert_state.shape[1]
    A = expert_action.shape[1]

    X = np.concatenate((expert_state, unknown_state)) + np.random.normal(0, 1e-2, (N_EXPERT+N_UNKNOWN, S))
    y = np.concatenate((expert_action, unknown_action)) + np.random.normal(0, 1e-2, (N_EXPERT+N_UNKNOWN, A))
    X = normalize(X)
    y = normalize(y)

    lgpr = LGPR(X,y)
    lgpr.init_kernel()
    lev = lgpr.solve()
    print(lev)

    lev = lev.tolist()
    save_file_name = "expert_lev"
    save_file = DATA_PATH + save_file_name + ".pkl"
    with open(save_file, 'wb') as pk:
        pickle.dump(lev, pk)
    
