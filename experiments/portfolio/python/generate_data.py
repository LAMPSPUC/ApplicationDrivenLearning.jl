import os
import pyepo
import pandas as pd

from config import *


if __name__ == '__main__':
    # generate data
    cov, x, c = pyepo.data.portfolio.genData(
        num_data=n+2*n, 
        num_features=p,
        num_assets=m, 
        deg=deg
    )

    # transform data into dataframes
    cov_df = pd.DataFrame(
        cov, 
        index=[f'asset_{i+1}' for i in range(m)],
        columns=[f'asset_{i+1}' for i in range(m)]
    )
    x_df = pd.DataFrame(x, columns=[f'feat_{i}' for i in range(1, p+1)])
    c_df = pd.DataFrame(c, columns=[f'asset_{i}' for i in range(1, m+1)])

    # store in csv files
    cov_df.to_csv(os.path.join(INPT_PATH, 'cov.csv'), index=False)
    x_df.to_csv(os.path.join(INPT_PATH, 'x.csv'), index=False)
    c_df.to_csv(os.path.join(INPT_PATH, 'c.csv'), index=False)
