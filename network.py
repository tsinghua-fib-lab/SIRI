import torch
from torch import nn
import torch.nn.functional as F

class p_t_z(nn.Module):
    def __init__(self, dim_latent, n_hidden_dec, dim_hidden_dec, n_t):
        super(p_t_z, self).__init__()
        self.n_hidden = n_hidden_dec
        self.input_net = nn.Linear(dim_latent, dim_hidden_dec)
        self.hidden_net = nn.ModuleList([nn.Linear(dim_hidden_dec, dim_hidden_dec) for i in range(n_hidden_dec - 1)])
        self.treatment_net = nn.Linear(dim_hidden_dec, n_t)
    def forward(self, z):
        z = F.elu(self.input_net(z))
        for i in range(self.n_hidden - 1):
            z = F.elu(self.hidden_net[i](z))
        t = self.treatment_net(z)
        return t

class q_z_t(nn.Module):
    def __init__(self, dim_latent, n_hidden_enc, dim_hidden_enc, n_t):
        super(q_z_t, self).__init__()
        self.n_hidden = n_hidden_enc
        self.dim_latent = dim_latent
        self.input_net_t = nn.Linear(n_t, dim_hidden_enc)
        self.hidden_net_t = nn.ModuleList([nn.Linear(dim_hidden_enc, dim_hidden_enc) for i in range(n_hidden_enc - 1)])
        self.zt_net_loc = nn.Linear(dim_hidden_enc, dim_latent)
        self.zt_net_log_std = nn.Linear(dim_hidden_enc, dim_latent)

    def forward(self, t, state='train'):
        zt = F.elu(self.input_net_t(t))
        for i in range(self.n_hidden - 1):
            zt = F.elu(self.hidden_net_t[i](zt))
        if state == 'pretrain':
            zt_loc = self.zt_net_loc(zt)
            return zt_loc
        if state == 'train':
            zt_loc = self.zt_net_loc(zt)
            zt_log_std = self.zt_net_log_std(zt)
            return zt_loc, zt_log_std

class DomainClassifer(nn.Module):
    def __init__(self, dim_x, dim_z, n_hidden, dim_hidden):
        super(DomainClassifer, self).__init__()
        self.n_hidden = n_hidden
        self.input_net = nn.Linear(dim_x + dim_z, dim_hidden)
        self.hidden_net = nn.ModuleList([nn.Linear(dim_hidden, dim_hidden) for i in range(n_hidden - 1)])
        self.output_net = nn.Linear(dim_hidden, 1)
    def forward(self, x, z):
        zx = torch.cat([x, z], 1)
        zy = F.elu(self.input_net(zx))
        for i in range(self.n_hidden - 1):
            zy = F.elu(self.hidden_net[i](zy))
        y = torch.sigmoid(self.output_net(zy))
        return y