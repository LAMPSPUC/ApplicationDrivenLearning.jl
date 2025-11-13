import os
import random
import torch
import numpy as np

# fix random seed
random.seed(135)
np.random.seed(135)
torch.manual_seed(135)

# parameters
m = 50 # number of assets
n = 100 # number of data
p = 5 # size of feature
deg = 6 # polynomial degree
cur_dir = "/".join(__file__.split(r"/")[:-1])
DATA_PATH = os.path.join(cur_dir, "..//data")
INPT_PATH = os.path.join(DATA_PATH, "input")
OUTP_PATH = os.path.join(DATA_PATH, "pyepo_result")
