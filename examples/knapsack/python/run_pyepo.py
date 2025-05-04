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


# build linear model
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(p, m)

    def forward(self, x):
        out = self.linear(x)
        return out

def trainModel(reg, loss_func, method_name, num_epochs=40, lr=1e-1):
    # set adam optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # train mode
    reg.train()
    # log
    loss_log, regret_log = [], [pyepo.metric.regret(reg, optmodel, loader_test)]
    for epoch in range(num_epochs):
        # load data
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            # cuda
            if torch.cuda.is_available():
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()
            # forward pass
            cp = reg(x)
            if method_name == "spo+":
                loss = loss_func(cp, c, w, z)
            elif method_name in ["ptb", "pfy", "imle", "nce"]:
                loss = loss_func(cp, w)
            elif method_name in ["dbb", "nid"]:
                loss = loss_func(cp, c, z)
            elif method_name in ["2s", "pg", "ltr"]:
                loss = loss_func(cp, c)
            # record loss
            loss_log.append(loss.item())
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # record regrer
        if epoch % 2 == 0:
            regret = pyepo.metric.regret(reg, optmodel, loader_test)
            # print("Epoch {:2},  Loss: {:9.4f},  Regret: {:7.4f}%".format(epoch, loss.item(), regret*100))
            regret_log.append(regret)


if __name__ == '__main__':
    
    # load data from files
    weights_df = pd.read_csv(os.path.join(INPT_PATH, 'weights.csv'))
    x_df = pd.read_csv(os.path.join(INPT_PATH, 'x.csv'))
    c_df = pd.read_csv(os.path.join(INPT_PATH, 'c.csv'))
    weights = weights_df.values.T
    x = x_df.values
    c = c_df.values

    # init optimization model
    optmodel = pyepo.model.grb.knapsackModel(weights, caps)

    # split data and get dataloaders
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=2*n, random_state=246, shuffle=False)
    dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)
    loader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # train model
    reg = LinearRegression()
    spop = pyepo.func.SPOPlus(optmodel, processes=1)
    trainModel(reg, loss_func=spop, method_name="spo+")

    # get test solutions and costs
    predicted_costs = []
    solutions = []
    costs = []
    for i, data in enumerate(loader_test):
        x, c, w, z = data
        cp = reg(x)
        cp = cp.to("cpu").detach().numpy()
        optmodel.setObj(cp[0])
        wp, iter_cost = optmodel.solve()

        # get assessed cost
        c = c.to("cpu").detach().numpy()
        iter_cost = c[0].dot(wp)

        predicted_costs.append(cp)
        solutions.append(wp)
        costs.append(iter_cost)

    predicted_costs = np.concatenate(predicted_costs)
    solutions = np.concatenate([solutions])
    costs = np.array(costs)

    predicted_costs_df = pd.DataFrame(predicted_costs, columns=[f'item_{i}' for i in range(1, m+1)])
    solutions_df = pd.DataFrame(solutions.astype(int), columns=[f'item_{i}' for i in range(1, m+1)])
    costs_df = pd.DataFrame(costs, columns=["cost"])

    # store results in csv files
    predicted_costs_df.to_csv(os.path.join(OUTP_PATH, 'predictions.csv'), index=False)
    solutions_df.to_csv(os.path.join(OUTP_PATH, 'solutions.csv'), index=False)
    costs_df.to_csv(os.path.join(OUTP_PATH, 'costs.csv'), index=False)
