import os
import random
import pyepo
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config import *


if __name__ == '__main__':
    # generate data
    x, c = pyepo.data.shortestpath.genData(
        n+2*n, p, 
        grid=grid, deg=deg, 
        noise_width=noise_width
    )

    # transform data into dataframes
    x_df = pd.DataFrame(x, columns=[f'feat_{i}' for i in range(1, p+1)])
    c_df = pd.DataFrame(c, columns=[f'edge_{i}' for i in range(1, m+1)])

    # store in csv files
    x_df.to_csv(os.path.join(INPT_PATH, 'x.csv'), index=False)
    c_df.to_csv(os.path.join(INPT_PATH, 'c.csv'), index=False)
