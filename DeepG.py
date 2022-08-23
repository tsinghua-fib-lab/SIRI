from __future__ import print_function
from importlib.machinery import SourceFileLoader
import argparse
from json import decoder
from re import L
from typing import no_type_check_decorator
import torch.optim as optim
import torch.utils.data.distributed
import pandas as pd
import numpy as np
import random
import os
import time
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from models.decompose import X_Encoder, X_Regressor
from network import DomainClassifer
from torch import nn


def Decomposed_pred(args, train_data, test_data, flow_df, oa2features, od2flow, oa2centroid, historyflow, treatment_dict, t_vae_model=None):
    def train_state():
        dc.eval()
        Encoder_a.train()
        Encoder_b.train()
        Encoder_c.train()
        Map_a_b_T.train()
        Map_b_c_Y.train()
        Map_a_y.eval()
        Map_c_T.eval()
        Decoder.train()

    def train_state_dis():
        dc.eval()
        Encoder_a.eval()
        Encoder_b.eval()
        Encoder_c.eval()
        Map_a_y.train()
        Map_a_b_T.eval()
        Map_c_T.train()
        Map_b_c_Y.eval()
        Decoder.eval()

    def train_state_cls():
        dc.train()
        Encoder_a.eval()
        Encoder_b.eval()
        Encoder_c.eval()
        Map_a_y.train()
        Map_a_b_T.eval()
        Map_c_T.train()
        Map_b_c_Y.eval()
        Decoder.eval()

    def eval_state():
        dc.eval()
        Encoder_a.eval()
        Encoder_b.eval()
        Encoder_c.eval()
        Map_a_y.eval()
        Map_a_b_T.eval()
        Map_c_T.eval()
        Map_b_c_Y.eval()
        Decoder.eval()

    def calcSampleWeights(X, T, q_z_t_dist, lat_clf, args):
        q_z_t_dist.eval()
        lat_clf.eval()
        nums = 100
        z_batch_loc, z_batch_log_std = q_z_t_dist(T)
        for j in range(nums):
            z_batch = z_batch_loc + torch.exp(z_batch_log_std) * torch.randn(size=z_batch_loc.size()).to(args.device)
            pre_d = lat_clf(X, z_batch).cpu().detach().numpy().squeeze()
            if j == 0:
                weight = ((1 - pre_d) / pre_d) / nums
            else:
                weight += ((1 - pre_d) / pre_d) / nums
        weight = 1 / weight
        return weight

    def decompose(train_dataset, valid_dataset):
        patience, n_step, t_step, d_step, p_step = 0, 0, 0, 0, 0
        min_loss = 1e15
        n_train, n_valid = len(train_dataset), len(valid_dataset)
        # Learn the initial causal disentangled representation
        for epoch in range(1, 101):
            train_state()

            running_loss, loss1sum, loss2sum, loss3sum, loss4sum, reconlosssum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for batch_idx, data_temp in enumerate(train_loader):
                b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                ori_target = torch.flatten(ori_target, 0, 2)

                optimizer.zero_grad()
                optimizer_dec.zero_grad()
                if args.dvs == 'gpu':
                    data, treatment, target, ori_target = data.to(args.device), treatment.to(args.device), target.to(
                        args.device), ori_target.to(args.device)

                X_a = Encoder_a.forward(data)
                X_b = Encoder_b.forward(data)
                X_c = Encoder_c.forward(data)
                X_a_y = Map_a_y.forward(X_a)
                X_a_b_T = Map_a_b_T.forward(torch.cat([X_a, X_b], dim=1))
                X_c_t = Map_c_T.forward(X_c)
                output = Map_b_c_Y.forward(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0))
                rec = Decoder.forward(torch.cat([X_a, X_b, X_c], dim=1))
                weight = torch.ones((data.shape[0], 1)).to(args.device)
                loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                loss_2 = Map_a_b_T.loss(X_a_b_T, treatment) * args.beta_b
                loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                loss_4 = Map_b_c_Y.loss(output, target, weight) * args.beta_d
                recon_loss = Decoder.loss(rec, data)
                loss = loss_4 - loss_3 + loss_2 - loss_1 + recon_loss
                if epoch >= 20:
                    loss = loss_4 - loss_3 + loss_2 - loss_1 + recon_loss
                else:
                    loss = loss_4 + loss_2 + recon_loss
                loss.backward()
                optimizer.step()
                optimizer_dec.step()
                running_loss += loss.item()
                loss1sum += loss_1.item()
                loss2sum += loss_2.item()
                loss3sum += loss_3.item()
                loss4sum += loss_4.item()
                reconlosssum += recon_loss.item()

            print('Train Epoch: {} [{}/{} \tLoss: {:.6f}'.format(epoch, batch_idx * len(b_data), len(train_loader),
                                                                 running_loss / n_train))
            print('Recon_loss: {:.3f} Loss_a2Y: {:.3f} Loss_ab2T: {:.3f} Loss_c2T: {:.3f} Loss_bc2Y: {:.3f}'.format(
                reconlosssum / n_train, loss1sum / n_train, loss2sum / n_train, loss3sum / n_train, loss4sum / n_train))
            mlflow.log_metric(key='DG_loss', value=running_loss / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss1', value=loss1sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss2', value=loss2sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss3', value=loss3sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss4', value=loss4sum / n_train, step=n_step)
            mlflow.log_metric(key='Recon_loss', value=reconlosssum / n_train, step=n_step)
            n_step += 1

            # validation
            if epoch % 5 == 0 and epoch >= 20:
                eval_state()
                running_loss, loss1sum, loss2sum, loss3sum, loss4sum, reconlosssum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                n_origins = 0
                targets = []
                outputs = []
                for batch_idx, data_temp in enumerate(valid_loader):
                    b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                    ori_target = b_target
                    b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                    data = torch.flatten(b_data, 0, 2)
                    treatment = torch.flatten(b_treatment, 0, 2)
                    target = torch.flatten(b_target, 0, 2)
                    ori_target = torch.flatten(ori_target, 0, 2)

                    loss = 0.0
                    if args.dvs == 'gpu':
                        data, treatment, target, ori_target = data.to(args.device), treatment.to(
                            args.device), target.to(
                            args.device), ori_target.to(args.device)

                    X_a = Encoder_a.forward(data)
                    X_b = Encoder_b.forward(data)
                    X_c = Encoder_c.forward(data)
                    X_a_y = Map_a_y.forward(X_a)
                    X_a_b_T = Map_a_b_T.forward(torch.cat([X_a, X_b], dim=1))
                    X_c_t = Map_c_T.forward(X_c)
                    output = Map_b_c_Y.forward(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0))
                    rec = Decoder.forward(torch.cat([X_a, X_b, X_c], dim=1))
                    _, predicted_outcome = Map_b_c_Y.get_cpc(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0),
                                                             ori_target)
                    n_origins += 1
                    weight = torch.ones((data.shape[0], 1)).to(args.device)
                    loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                    loss_2 = Map_a_b_T.loss(X_a_b_T, treatment) * args.beta_b
                    loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                    loss_4 = Map_b_c_Y.loss(output, target, weight) * args.beta_d
                    recon_loss = Decoder.loss(rec, data)
                    loss = loss_4 - loss_3 + loss_2 - loss_1 + recon_loss
                    running_loss += loss.item()
                    loss1sum += loss_1.item()
                    loss2sum += loss_2.item()
                    loss3sum += loss_3.item()
                    loss4sum += loss_4.item()
                    reconlosssum += recon_loss.item()
                    observed = list(ori_target.squeeze().cpu().detach().numpy())
                    targets += observed
                    outputs += list(predicted_outcome.cpu().detach().numpy())

                print('-----------------------------Validation Stage-----------------------')
                mse = mean_squared_error(outputs, targets)
                mae = mean_absolute_error(outputs, targets)
                mlflow.log_metric(key='Validation_MSE', value=mse, step=t_step)
                mlflow.log_metric(key='Validation_MAE', value=mae, step=t_step)
                mlflow.log_metric(key='DG_validation_loss', value=running_loss / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss1', value=loss1sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss2', value=loss2sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss3', value=loss3sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss4', value=loss4sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_recon_loss', value=reconlosssum / n_valid, step=t_step)
                print('Average loss, MSE, MAE of vali tiles: {}, {}, {}'.format(running_loss / n_valid, mse, mae))
                t_step += 1

                loss_4_mean = loss4sum / n_valid
                model_type = 'DG'
                if loss_4_mean < min_loss:
                    patience = 0
                    min_loss = loss_4_mean
                    mlflow.log_metric(key='best_epoch', value=epoch)
                    mlflow.log_metric(key='min_loss_4', value=min_loss)
                    fname = './results/Encoder_a_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_a.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Encoder_b_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_b.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Encoder_c_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_c.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Map_b_c_Y_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Map_b_c_Y.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Dec_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Decoder.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)

                if epoch % 5 == 0:
                    train_state_dis()
                    print('-------------------------------Discriminator Training Stage---------------------------')
                    for ep in range(5):
                        discriminator_loss, loss1sum, loss2sum, loss3sum, loss4sum = 0.0, 0.0, 0.0, 0.0, 0.0
                        for batch_idx, data_temp in enumerate(train_loader):
                            b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                            ori_target = b_target
                            b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                            data = torch.flatten(b_data, 0, 2)
                            treatment = torch.flatten(b_treatment, 0, 2)
                            target = torch.flatten(b_target, 0, 2)
                            ori_target = torch.flatten(ori_target, 0, 2)

                            optimizer_dis.zero_grad()
                            loss = 0.0
                            if args.dvs == 'gpu':
                                data, treatment, target, ori_target = data.to(args.device), treatment.to(
                                    args.device), target.to(args.device), ori_target.to(args.device)

                            X_a = Encoder_a.forward(data)
                            X_c = Encoder_c.forward(data)
                            X_a_y = Map_a_y.forward(X_a)
                            X_c_t = Map_c_T.forward(X_c)
                            loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                            loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                            loss = loss_1 + loss_3
                            loss.backward()
                            optimizer_dis.step()
                            discriminator_loss += loss.item()
                            loss1sum += loss_1.item()
                            loss3sum += loss_3.item()

                        print('Discriminator Training Epoch: {} Loss: {:.3f}'.format(ep, discriminator_loss / n_train))
                        print('Loss_a2Y: {:.3f} Loss_c2T: {:.3f}'.format(loss1sum / n_train, loss3sum / n_train))
                        mlflow.log_metric(key='Discriminator_loss', value=discriminator_loss / n_train, step=p_step)
                        mlflow.log_metric(key='DG_loss1', value=loss1sum / n_train, step=p_step)
                        mlflow.log_metric(key='DG_loss3', value=loss3sum / n_train, step=p_step)
                        p_step += 1
                    print('-------------------------------Training Stage--------------------------')
        return n_step, t_step, d_step, p_step

    def reweight(t_vae_model, n_step, t_step, d_step, p_step ):
        patience = 0
        min_loss = 1e15
        n_train, n_valid = len(train_dataset), len(valid_dataset)
        bceloss = nn.BCELoss(reduction='none')
        # Reweighting module: train classifier
        min_loss = 1e15
        train_state_cls()
        min_c_loss = 1e10
        for ep in range(100):
            cl_losses = []
            for batch_idx, data_temp in enumerate(train_loader):
                b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                ori_target = torch.flatten(ori_target, 0, 2)
                loss = 0.0
                if args.dvs == 'gpu':
                    data, treatment = data.to(args.device), treatment.to(args.device)
                    X_b = Encoder_b.forward(data)
                    z_batch_loc, z_batch_log_std = t_vae_model(treatment)
                    z_batch = z_batch_loc + torch.exp(z_batch_log_std) * torch.randn(size=z_batch_loc.size()).to(
                        args.device)
                    z_batch_neg = torch.randn(size=z_batch.size()).to(args.device)

                    x_batch = torch.cat([X_b, X_b], dim=0)
                    z_batch = torch.cat([z_batch, z_batch_neg], dim=0)
                    label_batch = torch.cat([torch.zeros(data.shape[0], 1), torch.ones(data.shape[0], 1)],
                                            dim=0).to(
                        args.device)

                    pre_d = dc(x_batch, z_batch)
                    optimizer_dc.zero_grad()
                    loss = bceloss(pre_d, label_batch).sum()
                    loss.backward()
                    optimizer_dc.step()
                    cl_losses.append(loss.cpu().detach().numpy())
            if sum(cl_losses) / len(train_loader) < min_c_loss:
                min_c_loss = sum(cl_losses) / len(train_loader)
                torch.save(dc, 'models/DC_' + args.model_name_suffix + '.mdl', _use_new_zipfile_serialization=False)
            mlflow.log_metric(key='classifier_loss', value=sum(cl_losses) / len(train_loader), step=d_step)
            print('Classifier Epoch %d' % ep)
            print('Classifier Loss %f' % (sum(cl_losses) / len(train_loader)))
            d_step += 1
        dc_trained = torch.load('models/DC_' + args.model_name_suffix + '.mdl')

        # Reweighting module: calculate sample weights
        weights_train, weights_valid = [], []
        eval_state()
        with torch.no_grad():
            for batch_idx, data_temp in enumerate(train_loader):
                b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                ori_target = torch.flatten(ori_target, 0, 2)
                if args.dvs == 'gpu':
                    data, treatment, target, ori_target = data.to(args.device), treatment.to(
                        args.device), target.to(
                        args.device), ori_target.to(args.device)
                X_b = Encoder_b.forward(data)
                weight = calcSampleWeights(X_b, treatment, t_vae_model, dc_trained, args)
                weights_train.append(weight)
            for batch_idx, data_temp in enumerate(valid_loader):
                b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                ori_target = torch.flatten(ori_target, 0, 2)
                if args.dvs == 'gpu':
                    data, treatment, target, ori_target = data.to(args.device), treatment.to(
                        args.device), target.to(
                        args.device), ori_target.to(args.device)
                X_b = Encoder_b.forward(data)
                weight = calcSampleWeights(X_b, treatment, t_vae_model, dc_trained, args)
                weights_valid.append(weight)

        # normalize learned weights
        weights = np.array(weights_train + weights_valid)
        weights_train = np.array(weights_train) / weights.mean()
        weights_valid = np.array(weights_valid) / weights.mean()
        weights_train = torch.from_numpy(weights_train).float().unsqueeze(-1)
        weights_valid = torch.from_numpy(weights_valid).float().unsqueeze(-1)
        return weights_train, weights_valid, n_step, t_step, d_step, p_step

    def combined(train_dataset, valid_dataset, weights_train, weights_valid, n_step, t_step, d_step, p_step ):
        n_train, n_valid = len(train_dataset), len(valid_dataset)
        patience = 0
        min_loss = 1e15
        # Introduce learned weights and continue on training
        for epoch in range(101, args.epochs + 1):
            train_state()

            running_loss, loss1sum, loss2sum, loss3sum, loss4sum, reconlosssum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for batch_idx, data_temp in enumerate(train_loader):
                b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                if args.vsr == 0:
                    weight = torch.ones((data.shape[0], 1))
                else:
                    weight = weights_train[batch_idx]
                ori_target = torch.flatten(ori_target, 0, 2)

                optimizer.zero_grad()
                optimizer_dec.zero_grad()
                if args.dvs == 'gpu':
                    data, treatment, target, ori_target, weight = data.to(args.device), treatment.to(
                        args.device), target.to(args.device), ori_target.to(args.device), weight.to(args.device)

                X_a = Encoder_a.forward(data)
                X_b = Encoder_b.forward(data)
                X_c = Encoder_c.forward(data)
                X_a_y = Map_a_y.forward(X_a)
                X_a_b_T = Map_a_b_T.forward(torch.cat([X_a, X_b], dim=1))
                X_c_t = Map_c_T.forward(X_c)
                output = Map_b_c_Y.forward(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0))
                rec = Decoder.forward(torch.cat([X_a, X_b, X_c], dim=1))
                loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                loss_2 = Map_a_b_T.loss(X_a_b_T, treatment) * args.beta_b
                loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                loss_4 = Map_b_c_Y.loss(output, target, weight) * args.beta_d
                recon_loss = Decoder.loss(rec, data)
                if epoch >= 20:
                    loss = loss_4 - loss_3 + loss_2 - loss_1 + recon_loss
                else:
                    loss = loss_4 + loss_2 + recon_loss
                loss.backward()

                optimizer.step()
                optimizer_dec.step()
                running_loss += loss.item()
                loss1sum += loss_1.item()
                loss2sum += loss_2.item()
                loss3sum += loss_3.item()
                loss4sum += loss_4.item()
                reconlosssum += recon_loss.item()

            print('Train Epoch: {} [{}/{} \tLoss: {:.6f}'.format(epoch, batch_idx * len(b_data), len(train_loader),
                                                                 running_loss / n_train))
            print(
                '     Recon_loss: {:.3f} Loss_a2Y: {:.3f} Loss_ab2T: {:.3f} Loss_c2T: {:.3f} Loss_bc2Y: {:.3f}'.format(
                    reconlosssum / n_train, loss1sum / n_train, loss2sum / n_train, loss3sum / n_train,
                    loss4sum / n_train))
            mlflow.log_metric(key='DG_loss', value=running_loss / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss1', value=loss1sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss2', value=loss2sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss3', value=loss3sum / n_train, step=n_step)
            mlflow.log_metric(key='DG_loss4', value=loss4sum / n_train, step=n_step)
            mlflow.log_metric(key='Recon_loss', value=reconlosssum / n_train, step=n_step)
            n_step += 1

            # validation
            if epoch % 5 == 0 and epoch >= 20:
                eval_state()
                running_loss, loss1sum, loss2sum, loss3sum, loss4sum, reconlosssum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                n_origins = 0
                targets = []
                outputs = []
                valid_accuracy = 0
                for batch_idx, data_temp in enumerate(valid_loader):
                    b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                    ori_target = b_target
                    b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                    data = torch.flatten(b_data, 0, 2)
                    treatment = torch.flatten(b_treatment, 0, 2)
                    target = torch.flatten(b_target, 0, 2)
                    ori_target = torch.flatten(ori_target, 0, 2)
                    if args.vsr == 0:
                        weight = torch.ones((data.shape[0], 1))
                    else:
                        weight = weights_valid[batch_idx]

                    loss = 0.0
                    if args.dvs == 'gpu':
                        data, treatment, target, ori_target, weight = data.to(args.device), treatment.to(
                            args.device), target.to(args.device), ori_target.to(args.device), weight.to(
                            args.device)

                    X_a = Encoder_a.forward(data)
                    X_b = Encoder_b.forward(data)
                    X_c = Encoder_c.forward(data)
                    X_a_y = Map_a_y.forward(X_a)
                    X_a_b_T = Map_a_b_T.forward(torch.cat([X_a, X_b], dim=1))
                    X_c_t = Map_c_T.forward(X_c)
                    output = Map_b_c_Y.forward(
                        torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0))  # 没有输入treatment变量
                    rec = Decoder.forward(torch.cat([X_a, X_b, X_c], dim=1))
                    _, predicted_outcome = Map_b_c_Y.get_cpc(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0),
                                                             ori_target)
                    n_origins += 1
                    loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                    loss_2 = Map_a_b_T.loss(X_a_b_T, treatment) * args.beta_b
                    loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                    loss_4 = Map_b_c_Y.loss(output, target, weight) * args.beta_d
                    recon_loss = Decoder.loss(rec, data)
                    loss = loss_4 - loss_3 + loss_2 - loss_1 + recon_loss
                    running_loss += loss.item()
                    loss1sum += loss_1.item()
                    loss2sum += loss_2.item()
                    loss3sum += loss_3.item()
                    loss4sum += loss_4.item()
                    reconlosssum += recon_loss.item()
                    observed = list(ori_target.squeeze().cpu().detach().numpy())
                    targets += observed
                    outputs += list(predicted_outcome.cpu().detach().numpy())

                print('-----------------------------Validation Stage-----------------------')
                mse = mean_squared_error(outputs, targets)
                mae = mean_absolute_error(outputs, targets)
                mlflow.log_metric(key='Validation_MSE', value=mse, step=t_step)
                mlflow.log_metric(key='Validation_MAE', value=mae, step=t_step)
                mlflow.log_metric(key='DG_validation_loss', value=running_loss / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss1', value=loss1sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss2', value=loss2sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss3', value=loss3sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_loss4', value=loss4sum / n_valid, step=t_step)
                mlflow.log_metric(key='DG_validation_recon_loss', value=reconlosssum / n_valid, step=t_step)
                print('Average loss, MSE, MAE of vali tiles: {}, {}, {}'.format(running_loss / n_valid, mse, mae))
                t_step += 1

                loss_4_mean = loss4sum / n_valid
                model_type = 'DG'
                if loss_4_mean < min_loss:
                    patience = 0
                    min_loss = loss_4_mean
                    mlflow.log_metric(key='best_epoch', value=epoch)
                    mlflow.log_metric(key='min_loss_4', value=min_loss)
                    fname = './results/Encoder_a_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_a.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Encoder_b_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_b.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Encoder_c_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Encoder_c.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Map_b_c_Y_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Map_b_c_Y.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                    fname = './results/Dec_{}_{}.pt'.format(model_type, args.model_name_suffix)
                    torch.save({'model_state_dict': Decoder.state_dict(),
                                }, fname, _use_new_zipfile_serialization=False)
                else:
                    if epoch > 20:
                        patience += 1
                        if patience > args.patience:
                            break

                if epoch % 5 == 0:
                    train_state_dis()
                    print('-------------------------------Discriminator Training Stage---------------------------')
                    for ep in range(5):
                        discriminator_loss, loss1sum, loss2sum, loss3sum, loss4sum = 0.0, 0.0, 0.0, 0.0, 0.0
                        for batch_idx, data_temp in enumerate(train_loader):
                            b_data, b_treatment, b_target = data_temp[0], data_temp[1], data_temp[2]
                            ori_target = b_target
                            b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                            data = torch.flatten(b_data, 0, 2)
                            treatment = torch.flatten(b_treatment, 0, 2)
                            target = torch.flatten(b_target, 0, 2)
                            ori_target = torch.flatten(ori_target, 0, 2)

                            optimizer_dis.zero_grad()
                            loss = 0.0
                            if args.dvs == 'gpu':
                                data, treatment, target, ori_target = data.to(args.device), treatment.to(
                                    args.device), target.to(args.device), ori_target.to(args.device)

                            X_a = Encoder_a.forward(data)
                            X_c = Encoder_c.forward(data)
                            X_a_y = Map_a_y.forward(X_a)
                            X_c_t = Map_c_T.forward(X_c)
                            loss_1 = Map_a_y.loss(X_a_y, target) * args.beta_a
                            loss_3 = Map_c_T.loss(X_c_t, treatment) * args.beta_c
                            loss = loss_1 + loss_3
                            loss.backward()
                            optimizer_dis.step()
                            discriminator_loss += loss.item()
                            loss1sum += loss_1.item()
                            loss3sum += loss_3.item()

                        print('Discriminator Training Epoch: {} Loss: {:.3f}'.format(ep, discriminator_loss / n_train))
                        print('      Loss_a2Y: {:.3f} Loss_c2T: {:.3f}'.format(loss1sum / n_train, loss3sum / n_train))
                        mlflow.log_metric(key='Discriminator_loss', value=discriminator_loss / n_train, step=p_step)
                        mlflow.log_metric(key='DG_loss1', value=loss1sum / n_train, step=p_step)
                        mlflow.log_metric(key='DG_loss3', value=loss3sum / n_train, step=p_step)
                        p_step += 1
                    print('-------------------------------Training Stage--------------------------')

    def train(train_dataset, valid_dataset, t_vae_model):
        n_step, t_step, d_step, p_step  = decompose(train_dataset, valid_dataset)
        weights_train, weights_valid, n_step, t_step, d_step, p_step  = reweight(t_vae_model, n_step, t_step, d_step, p_step )
        combined(train_dataset, valid_dataset, weights_train, weights_valid, n_step, t_step, d_step, p_step )

    def evaluate():
        eval_state()
        loc2cpc_numerator = {}

        targets = []
        outputs = []
        n_test = len(test_loader)
        num = 0
        with torch.no_grad():
            for data_temp in test_loader:
                num += 1
                b_data, b_treatment, b_target, ids = data_temp[0], data_temp[1], data_temp[2], data_temp[3]
                ori_target = b_target
                b_target = b_target / torch.sum(b_target, axis=2).squeeze(dim=0)
                data = torch.flatten(b_data, 0, 2)
                treatment = torch.flatten(b_treatment, 0, 2)
                target = torch.flatten(b_target, 0, 2)
                ori_target = torch.flatten(ori_target, 0, 2)

                loss = 0.0
                if args.dvs == 'gpu':
                    data, treatment, target, ori_target = data.to(args.device), treatment.to(args.device), target.to(
                        args.device), ori_target.to(args.device)  # T和X要分开输入

                X_a = Encoder_a.forward(data)
                X_b = Encoder_b.forward(data)
                X_c = Encoder_c.forward(data)
                cpc, predicted_outcome = Map_b_c_Y.get_cpc(torch.cat([X_b, X_c, treatment], dim=1).unsqueeze(dim=0), ori_target,
                                           numerator_only=False)
                loc2cpc_numerator[ids[0][0]] = cpc
                observed = list(ori_target.squeeze().cpu().detach().numpy())
                targets += observed
                outputs += list(predicted_outcome.cpu().detach().numpy())

        cpc_df = pd.DataFrame(
            {'Block_id': list(loc2cpc_numerator.keys()), 'fenzi': [m[0] for m in list(loc2cpc_numerator.values())],
             'fenmu': [m[1] for m in list(loc2cpc_numerator.values())]})

        print('Average CPC of test tiles: {} '.format(cpc_df.fenzi.sum() / cpc_df.fenmu.sum()))  # 对tile做平均
        mlflow.log_metric(key='CPC_mean', value=cpc_df.fenzi.sum() / cpc_df.fenmu.sum())

        mse = mean_squared_error(outputs, targets)
        mae = mean_absolute_error(outputs, targets)
        print('Average MSE of test tiles: {}'.format(mse))
        print('Average MAE of test tiles: {}'.format(mae))
        mlflow.log_metric(key='test_mae', value=mae)
        mlflow.log_metric(key='test_mse', value=mse)

    model_type = 'DG'
    path = './data_loader.py'
    dgd = SourceFileLoader('dg_data', path).load_module()
    path = './utils.py'
    utils = SourceFileLoader('utils', path).load_module()

    o2d2flow = {}
    for (o, d), f in od2flow.items():
        try:
            d2f = o2d2flow[o]
            d2f[d] = f
        except KeyError:
            o2d2flow[o] = {d: f}
            
    all_data = train_data+test_data
    train_data, valid_data = train_test_split(train_data, train_size = 0.66, random_state=args.seed)

    train_dataset_args = {'o2d2flow': o2d2flow,
                        'oa2features': oa2features,
                        'oa2centroid': oa2centroid,
                        'dim_dests': 6000,
                        'frac_true_dest': 0.0,
                        'model': model_type,
                        'historyflow': historyflow, 
                        'treatmentdict': treatment_dict, 
                        'trainstate': True, 
                        'd_set': (train_data+valid_data), 
                        'flow_df': flow_df}
    valid_dataset_args = {'o2d2flow': o2d2flow,
                        'oa2features': oa2features,
                        'oa2centroid': oa2centroid,
                        'dim_dests': 6000,
                        'frac_true_dest': 0.0,
                        'model': model_type,
                        'historyflow': historyflow, 
                        'treatmentdict': treatment_dict, 
                        'trainstate': True, 
                        'd_set': (train_data+valid_data), 
                        'flow_df': flow_df}
    test_dataset_args = {'o2d2flow': o2d2flow,
                        'oa2features': oa2features,
                        'oa2centroid': oa2centroid,
                        'dim_dests': int(1e9),
                        'frac_true_dest': 0.0,
                        'model': model_type,
                        'historyflow': historyflow,
                        'treatmentdict': treatment_dict,
                        'trainstate': False, 
                        'd_set': test_data, 
                        'flow_df': flow_df}

    train_dataset = dgd.FlowDataset(train_data, **train_dataset_args)
    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.seed))

    valid_dataset = dgd.FlowDataset(valid_data, **valid_dataset_args)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, worker_init_fn=np.random.seed(args.seed))

    test_dataset = dgd.FlowDataset(all_data, **test_dataset_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=args.test_batch_size, shuffle=False, worker_init_fn=np.random.seed(args.seed))
    dim_X = len(train_dataset.get_features(train_data[0], train_data[0]))

    dim_input = args.dim_c + args.dim_b + len(list(treatment_dict.values())[0])*2
    dim_T = len(list(treatment_dict.values())[0])*2
    dim_classifier = args.dim_b
    dc = DomainClassifer(dim_classifier, args.vae_dim_latent, args.classifer_n_hidden, args.classifer_dimhidden)

    Encoder_a = X_Encoder(dim_X, args.dim_a)    # instrumental variable
    Encoder_b = X_Encoder(dim_X, args.dim_b)    # confounding variable
    Encoder_c = X_Encoder(dim_X, args.dim_c)    # adjusting variable
    Map_a_y = X_Regressor(args.dim_a, 1)
    Map_a_b_T = X_Regressor(args.dim_a+args.dim_b, dim_T) 
    Map_c_T = X_Regressor(args.dim_c, dim_T) 
    Map_b_c_Y = utils.instantiate_model(oa2centroid, oa2features, dim_input, device=args.device)
    Decoder = X_Regressor(args.dim_a+args.dim_b+args.dim_c, dim_X)

    if args.dvs == 'gpu':
        dc, Encoder_a, Encoder_b, Encoder_c, Map_a_y, Map_a_b_T, Map_c_T, Map_b_c_Y, Decoder = dc.to(args.device), Encoder_a.to(args.device), Encoder_b.to(args.device), Encoder_c.to(args.device), Map_a_y.to(args.device), Map_a_b_T.to(args.device), Map_c_T.to(args.device), Map_b_c_Y.to(args.device), Decoder.to(args.device)

    optimizer_dc = optim.RMSprop(dc.parameters(), lr=args.lr_classifier, momentum=args.momentum)
    optimizer = optim.RMSprop(list(Encoder_a.parameters())+list(Map_a_b_T.parameters())+list(Map_b_c_Y.parameters())+list(Encoder_b.parameters())+list(Encoder_c.parameters()), lr=args.lr, momentum=args.momentum)
    optimizer_dec = optim.RMSprop(Decoder.parameters(), lr=args.lr_dec, momentum=args.momentum)
    optimizer_dis = optim.RMSprop(list(Map_a_y.parameters())+list(Map_c_T.parameters()), lr=args.lr_dis, momentum=args.momentum)

    t0 = time.time()
    train(train_dataset, valid_dataset, t_vae_model)
    t1 = time.time()
    print("Total training time: %s seconds" % (t1 - t0))

    print('Computing the CPC on test set, loc2cpc_numerator ...')
    fname = './results/Encoder_a_{}_{}.pt'.format(model_type, args.model_name_suffix)
    Encoder_a.load_state_dict(torch.load(fname)['model_state_dict'])
    fname = './results/Encoder_b_{}_{}.pt'.format(model_type, args.model_name_suffix)
    Encoder_b.load_state_dict(torch.load(fname)['model_state_dict'])
    fname = './results/Encoder_c_{}_{}.pt'.format(model_type, args.model_name_suffix)
    Encoder_c.load_state_dict(torch.load(fname)['model_state_dict'])
    fname = './results/Map_b_c_Y_{}_{}.pt'.format(model_type, args.model_name_suffix)
    Map_b_c_Y.load_state_dict(torch.load(fname)['model_state_dict'])

    evaluate()