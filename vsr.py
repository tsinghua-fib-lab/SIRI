import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch import nn
from network import p_t_z, q_z_t, DomainClassifer
import mlflow
from mlflow.tracking import MlflowClient
import mlflow
import mlflow.sklearn
from importlib.machinery import SourceFileLoader

path = './data_loader.py'
dgd = SourceFileLoader('dg_data', path).load_module()

def trainVAE(t, n_t, args):
    n_step = 0
    t = np.array(t)
    n = t.shape[0]
    dim_latent = args.vae_dim_latent
    n_hidden_enc = args.vae_n_hidden_enc
    n_hidden_dec = args.vae_n_hidden_dec
    dim_hidden_enc = args.vae_dim_hidden_enc
    dim_hidden_dec = args.vae_dim_hidden_dec
    p_t_z_dist = p_t_z(dim_latent, n_hidden_dec, dim_hidden_dec, n_t).to(args.device)
    q_z_t_dist = q_z_t(dim_latent, n_hidden_enc, dim_hidden_enc, n_t).to(args.device)
    print_iter = 10
    epoch_num = args.vae_epochs
    batch_size = args.vae_batchsize

    optimizer_enc = optim.Adam(list(q_z_t_dist.parameters()), lr=args.lr_vae_enc)
    optimizer_dec = optim.Adam(list(p_t_z_dist.parameters()), lr=args.lr_vae_dec)
    mseloss = nn.MSELoss(reduction='none')
    losses = []
    min_loss = 1e10
    beta = args.beta
    patience = 0
    for epoch in range(epoch_num):
        idx = np.random.permutation(n)
        print_loss = []
        con_loss = []
        lat_loss = []
        for i in range(0, n, batch_size):
            start, end = i, min(i + batch_size, n)
            t_batch = torch.FloatTensor(t[idx[start:end]]).to(args.device)

            z_infer_loc, z_infer_log_std = q_z_t_dist(t_batch)
            std_z = torch.randn(size=z_infer_loc.size()).to(args.device)
            z_infer_sample = z_infer_loc + torch.exp(z_infer_log_std) * std_z

            t_infer_sample = p_t_z_dist(z_infer_sample)   
            construct_loss = mseloss(t_infer_sample, t_batch).sum(1)
            latent_loss = (-z_infer_log_std + 1 / 2 * (
                        torch.exp(z_infer_log_std * 2) + z_infer_loc * z_infer_loc - 1)).sum(1)
            loss = torch.mean(construct_loss + beta*latent_loss)

            print_loss.append(torch.sum(construct_loss + args.beta*latent_loss).cpu().detach().numpy())
            con_loss.append(torch.sum(construct_loss).cpu().detach().numpy())
            lat_loss.append(torch.sum(latent_loss).cpu().detach().numpy())

            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
        print('Epoch %d' % epoch)
        print('Loss %f' % (sum(print_loss) / n))
        print('Construct Loss %f' % (sum(con_loss) / n))
        print('Latent Loss %f' % (sum(lat_loss) / n))
        losses.append(sum(print_loss) / n)
        mlflow.log_metric(key='vae_con_loss', value=sum(con_loss) / n, step=n_step)
        mlflow.log_metric(key='vae_lat_loss', value=sum(lat_loss) / n, step=n_step)
        mlflow.log_metric(key='vae_total_loss', value=sum(print_loss) / n, step=n_step)
        if sum(print_loss) / n < min_loss:
            patience = 0
            min_loss = sum(print_loss) / n
            vaeName = args.vae_name
            torch.save(p_t_z_dist, vaeName + args.model_name_suffix + '_p_t_z.mdl', _use_new_zipfile_serialization = False)
            torch.save(q_z_t_dist, vaeName + args.model_name_suffix + '_q_z_t.mdl', _use_new_zipfile_serialization = False)
        else: 
            patience += 1
            if patience > args.patience:
                break
        n_step += 1
    p_t_z_dist = torch.load(vaeName + args.model_name_suffix + '_p_t_z.mdl')
    q_z_t_dist = torch.load(vaeName + args.model_name_suffix + '_q_z_t.mdl')
    mlflow.log_metric(key='vae_min_loss', value=min_loss)
    return q_z_t_dist, p_t_z_dist

def t_vae(args, train_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict):
    data = dgd.VSR_Dataset(train_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict)
    t, odpair_list = data.get_data()
    q_z_t_dist, p_t_z_dist = trainVAE(t, t.shape[1], args)
    return q_z_t_dist


