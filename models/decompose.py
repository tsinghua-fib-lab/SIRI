import json
from turtle import forward
import pandas as pd
# import geopandas as gpd
import shapely
import area
import numpy as np
import random
import torch
from zipfile import ZipFile
from ast import literal_eval
from torch import nn
import torch.nn.functional as F

class X_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.35):
        super(X_Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(output_dim, output_dim)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(output_dim, output_dim)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(output_dim, output_dim)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(p)

        self.linear5 = torch.nn.Linear(output_dim, output_dim)
    
    def forward(self, X):
        lin1 = self.linear1(X)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        return lin5

class X_Regressor(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.35):
        super(X_Regressor, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, output_dim)
        self.relu1 = torch.nn.LeakyReLU()
        self.dropout1 = torch.nn.Dropout(p)

        self.linear2 = torch.nn.Linear(output_dim, output_dim)
        self.relu2 = torch.nn.LeakyReLU()
        self.dropout2 = torch.nn.Dropout(p)

        self.linear3 = torch.nn.Linear(output_dim, output_dim)
        self.relu3 = torch.nn.LeakyReLU()
        self.dropout3 = torch.nn.Dropout(p)

        self.linear4 = torch.nn.Linear(output_dim, output_dim)
        self.relu4 = torch.nn.LeakyReLU()
        self.dropout4 = torch.nn.Dropout(p)

        self.linear5 = torch.nn.Linear(output_dim, output_dim)
    
    def forward(self, X):
        lin1 = self.linear1(X)
        h_relu1 = self.relu1(lin1)
        drop1 = self.dropout1(h_relu1)

        lin2 = self.linear2(drop1)
        h_relu2 = self.relu2(lin2)
        drop2 = self.dropout2(h_relu2)

        lin3 = self.linear3(drop2)
        h_relu3 = self.relu3(lin3)
        drop3 = self.dropout3(h_relu3)

        lin4 = self.linear4(drop3)
        h_relu4 = self.relu4(lin4)
        drop4 = self.dropout4(h_relu4)

        lin5 = self.linear5(drop4)
        return lin5

    def loss(self, out, vT):
        MSE = torch.nn.MSELoss(reduce=None)
        return MSE(out, vT).sum()

    
