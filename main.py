from __future__ import print_function

import argparse

import torch.optim as optim
import torch.utils.data.distributed

import pandas as pd
import numpy as np

import random

import os

import time
import argparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from loader import load_config
from vsr import t_vae
from DeepG import Decomposed_pred
from importlib.machinery import SourceFileLoader

import torch
import torch.nn as nn
from torch.nn import init
import pdb
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["MKL_NUM_THREADS"] = "90"
os.environ["NUMEXPR_NUM_THREADS"] = "90"
os.environ["OMP_NUM_THREADS"] = "90"
os.environ["OPENBLAS_NUM_THREADS"] = "90"

# Training settings
parser = argparse.ArgumentParser(description='SIRI')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--device', default='cuda:0',
                    help='which device to use')
parser.add_argument('--dvs', default='gpu',
                    help='whether to use gpu or cpu')
parser.add_argument('--config_file', default='config/config')
parser.add_argument('--repetitions', type=int, default=20)
parser.add_argument('--classifer_dimhidden', type=int, default=12)
parser.add_argument('--classifer_n_hidden',type=int, default=4)
parser.add_argument('--classifer_batchsize', type=int,default=20000)
parser.add_argument('--classifer_epochs',type=int, default=50)
parser.add_argument('--vae_dim_latent', type=int, default=6)
parser.add_argument('--vae_n_hidden_enc', type=int, default=4)
parser.add_argument('--vae_n_hidden_dec', type=int, default=4)
parser.add_argument('--vae_dim_hidden_enc', type=int, default=12)
parser.add_argument('--vae_dim_hidden_dec', type=int, default=12)
parser.add_argument('--vae_epochs', type=int, default=200)
parser.add_argument('--vae_batchsize', type=int, default=20000)
parser.add_argument('--vae_name', type=str, default='models/noise1')
parser.add_argument('--lr_classifier', type=float, default=1e-4)
parser.add_argument('--lr_vae_enc', type=float, default=1e-2)
parser.add_argument('--lr_vae_dec', type=float, default=1e-4)
parser.add_argument('--pretrain_epochs', type=int, default=20)
parser.add_argument('--use_pre_vae', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--test-batch-size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--lr_dis', type=float, default=5e-6)
parser.add_argument('--lr_dec', type=float, default=5e-6)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--beta', type=float, default=1)    # hyperparameter of the VAE in the re-weighting module
parser.add_argument('--beta_a', type=float, default=10)    # loss balancing coefficients
parser.add_argument('--beta_b', type=float, default=1)
parser.add_argument('--beta_c', type=float, default=10)
parser.add_argument('--beta_d', type=float, default=2)
parser.add_argument('--exp_name', type=str, default='test', help='exp_name')
parser.add_argument('--model_name_suffix', type=str, default='10', help='exp_name')
parser.add_argument('--vsr', type=int, default=1, help='whether to use the reweighting module')
parser.add_argument('--split_file', type=str, default='train_test_split_urban_detect_v2.npy', help='split file')
parser.add_argument('--weight_decay', type=float, default=1, help='whether to add a decay coefficient on the learned weights')
parser.add_argument('--dim_a', type=int, default=20, help='instrumental variable')
parser.add_argument('--dim_b', type=int, default=25, help='confounding variable')
parser.add_argument('--dim_c', type=int, default=20, help='adjusting variable')

args = parser.parse_args()

if args.model_name_suffix == '10':  
    args.model_name_suffix = ''.join(random.sample(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e'], 8))

# random seeds
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)
    
def read_data():
    flow_df = pd.read_csv('flow_df.csv', sep=',')
    flow_df['residence'] = flow_df['residence'].apply(str)
    flow_df['workplace'] = flow_df['workplace'].apply(str)

    history_flow = pd.read_csv('history_flow.csv', sep=',')
    history_flow['residence'] = history_flow['residence'].apply(str)
    history_flow['workplace'] = history_flow['workplace'].apply(str)

    df = pd.read_csv('oa2features1.csv',sep=',')
    df['KEYS'] = df.apply(lambda x: str(x.KEYS),axis=1)
    df['VALUES'] = df.apply(lambda x: eval(x.VALUES),axis=1)
    keys = list(df['KEYS'])
    values = list(df['VALUES'])
    max_value = np.max(values, axis=0)
    min_value = np.min(values, axis=0)
    values = (values-min_value)/(max_value-min_value)
    values = [list(m) for m in values]    
    oa2features = dict(zip(keys, values))

    df = pd.read_csv('od2flow.csv',sep=',')
    df['VALUES'] = df.apply(lambda x: int(x.VALUES),axis=1)
    df['KEYS'] = df.apply(lambda x: eval(x.KEYS),axis=1)
    keys = list(df['KEYS'])
    values = list(df['VALUES'])
    od2flow = dict(zip(keys, values))

    df = pd.read_csv('oa2centroid.csv',sep=',')
    df['KEYS'] = df.apply(lambda x: str(x.KEYS),axis=1)
    df['VALUES'] = df.apply(lambda x: eval(x.VALUES),axis=1)
    keys = list(df['KEYS'])
    values = list(df['VALUES'])
    oa2centroid = dict(zip(keys, values))

    df = pd.read_csv('treatment_dict.csv',sep=',')
    df['KEYS'] = df.apply(lambda x: str(x.KEYS),axis=1)
    df['VALUES'] = df.apply(lambda x: eval(x.VALUES),axis=1)
    keys = list(df['KEYS'])
    values = list(df['VALUES'])
    max_value = np.max(values, axis=0)
    min_value = np.min(values, axis=0)
    values = (values-min_value)/(max_value-min_value)
    values = [list(m) for m in values]

    treatment_dict = dict(zip(keys, values))   
    return flow_df, history_flow, oa2features, od2flow, oa2centroid, treatment_dict

with mlflow.start_run():
    if args.dvs == 'gpu':
        torch.cuda.manual_seed(args.seed)
        args.device = args.device
    else:
        args.device = torch.device("cpu")

    mlflow.log_params(vars(args))
    # load data
    # flow_df: three columns: residence, workplace, flow (o, d, flow)   (only save od pairs with flow>0)
    # historyflow: four columns: residence, workplace, flow, odkey (o, d, flow, (o, d))   (only save od pairs with history flow>0)
    # oa2features: [o, features]
    # od2flow: [(o, d), flow] (all o, d pairs that we want to learn or predict even the flow is zero)
    # oa2centroid: [o, geographic centroid]
    # treatment_dict: [o, treatment(planned_development)]
    flow_df, historyflow, oa2features, od2flow, oa2centroid, treatment_dict = read_data()

    # split the origins into training data and testing data (Training data: developed regions; Testing data: developing regions)
    [train_data, test_data] = np.load(args.split_file, allow_pickle=True)
    if args.vsr == 1:
        t_vae_model = t_vae(args, train_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict)   # train the VAE of the re-weighting module
        Decomposed_pred(args, train_data, test_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict, t_vae_model)
    else:
        Decomposed_pred(args, train_data, test_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict)

 
