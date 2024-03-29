"""
Learned models to test metrics with.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import time
from torch.nn.utils import weight_norm

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FullyConvEncoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, bn=True, 
                 drop=False, nl=nn.ReLU(), stochastic=True, img_dim=64):
        super(FullyConvEncoderVAE, self).__init__()
        self.stochastic = stochastic
        self.layers = nn.ModuleList()
        self.latent_size = latent_size
        
        self.layers.append(nn.Conv2d(input, 32, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(32, 64, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(64, 128, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(128, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)
        
        self.layers.append(nn.Conv2d(128, 256, 4, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(256, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if img_dim == 64:
            n_size = 256 * 2 * 2
        elif img_dim == 128:
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        if self.stochastic:
            self.fc_mu = nn.Linear(n_size, latent_size)
            self.fc_logvar = nn.Linear(n_size, latent_size)
        else:
            self.fc = nn.Linear(n_size, latent_size)

        self.flatten = Flatten()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.flatten(x)

        if self.stochastic:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)

            # Reparameterize
            std = torch.exp(logvar / 2.0)

            eps = torch.randn_like(std)
            z = mu + eps * std

            return z, mu, logvar
        else: 
            return self.fc(x)


class FullyConvDecoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, output_nl=nn.Tanh(), bn=True, 
                    drop=False, nl=nn.ReLU(), img_dim=64):
        super(FullyConvDecoderVAE, self).__init__()
        self.bn = bn
        self.drop = drop
        self.layers = nn.ModuleList()

        if img_dim == 64:
            n_size = 256 * 2 * 2
        elif img_dim == 128:
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        self.layers.append(nn.ConvTranspose2d(n_size, 128, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(128, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose2d(128, 64, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if img_dim == 64:
            self.layers.append(nn.ConvTranspose2d(64, 32, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, input, 6, stride=2, bias=False))
        elif img_dim == 128:
            self.layers.append(nn.ConvTranspose2d(64, 32, 5, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, 16, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(16, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))

            self.layers.append(nn.ConvTranspose2d(16, input, 6, stride=2, bias=False))
        else:
            raise NotImplementedError()

        if output_nl != None:
            self.layers.append(output_nl)

        self.linear = nn.Linear(latent_size, n_size, bias=False)
        self.batchn = nn.BatchNorm1d(n_size, track_running_stats=True)
        self.dropout = nn.Dropout(p=0.5)
        self.nl = nl
        
    def forward(self, x):
        if self.bn:
            x = self.nl(self.batchn(self.linear(x)))
        elif self.drop:
            x = self.nl(self.dropout(self.linear(x)))
        else:
            x = self.nl(self.linear(x))

        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class FCNEncoderVAE(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        bn=False, 
        drop=False, 
        nl=nn.ReLU(), 
        hidden_size=256, 
        stochastic=True
    ):
        super(FCNEncoderVAE, self).__init__()
        self.flatten = Flatten()
        self.stochastic = stochastic
        self.bn = bn
        self.layers = nn.ModuleList()

        self.layers.append(torch.nn.Linear(dim_in, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if stochastic:
            self.layers.append(torch.nn.Linear(hidden_size, 2 * dim_out))
        else:
            self.layers.append(torch.nn.Linear(hidden_size, dim_out))

    def forward(self, x):
        x = self.flatten(x)
        for l in self.layers:
            x = l(x)

        if self.stochastic:
            mu, logvar = x.chunk(2, dim=1)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else: 
            return x


class FCNDecoderVAE(nn.Module):
    def __init__(self, dim_in, dim_out, bn=False, drop=False, nl=nn.ReLU(), output_nl=None, hidden_size=800):
        super(FCNDecoderVAE, self).__init__()
        self.dim_out = dim_out
        self.layers = nn.ModuleList()

        self.layers.append(torch.nn.Linear(dim_in, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        if bn: self.layers.append(nn.BatchNorm1d(hidden_size, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Linear(hidden_size, int(np.product(dim_out))))
        if output_nl != None:
            self.layers.append(output_nl)
            
    def forward(self, z):
        for l in self.layers:
            z = l(z)
        x = z.view(-1, self.dim_out)
        return x


class CNNEncoder1D(nn.Module):
    def __init__(self, input=6, latent_size=12, bn=True,
        drop=False, nl=nn.ReLU(), stochastic=True, kernel_size=3, datalength=32):
        super(CNNEncoder1D, self).__init__()
        self.stochastic = stochastic
        self.layers = nn.ModuleList()
        # 1D CNN (batch, channels, length)

        self.layers.append(torch.nn.Conv1d(input, 32, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(32, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Conv1d(32, 64, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Conv1d(64, 128, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(128, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(torch.nn.Conv1d(128, 256, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(256, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        n_size = 256 * datalength

        if self.stochastic:
            self.fc_mu = nn.Linear(n_size, latent_size)
            self.fc_logvar = nn.Linear(n_size, latent_size)
        else:
            self.fc = nn.Linear(n_size, latent_size)
        self.flatten = Flatten()

    def forward(self, x):

        for l in self.layers:
            x = l(x)

        x = self.flatten(x)
        if self.stochastic:
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else: 
            return self.fc(x)


class CNNDecoder1D(nn.Module):
    def __init__(self, input=6, latent_size=12, bn=True, 
        drop=False, nl=nn.ReLU(), output_nl=nn.Tanh(), kernel_size=3, datalength=32):
        super(CNNDecoder1D, self).__init__()
        self.bn = bn
        self.drop = drop
        self.layers = nn.ModuleList()

        self.datalength = datalength
        n_size = 256 * datalength

        self.layers.append(nn.ConvTranspose1d(256, 128, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(128, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose1d(128, 64, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose1d(64, 32, kernel_size, stride=1, padding=1, bias=False))
        if bn: self.layers.append(nn.BatchNorm1d(32, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose1d(32, input, kernel_size, stride=1, padding=1, bias=False))

        if output_nl != None:
            self.layers.append(output_nl)

        self.linear = nn.Linear(latent_size, n_size, bias=False)
        self.batchn = nn.BatchNorm1d(n_size, track_running_stats=True)
        self.dropout = nn.Dropout(p=0.5)
        self.nl = nl
        
    def forward(self, x):
        if self.bn:
            x = self.nl(self.batchn(self.linear(x)))
        elif self.drop:
            x = self.nl(self.dropout(self.linear(x)))
        else:
            x = self.nl(self.linear(x))

        x = x.reshape(-1, 256, self.datalength)
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x


class RNNEncoder(nn.Module):
    def __init__(
        self, 
        dim_in, 
        dim_out, 
        hidden_size=256,
        net_type="gru",
        train_initial_hidden=False,
        stochastic=False
    ):
        self.train_initial_hidden = train_initial_hidden
        self.net_type = net_type
        super(RNNEncoder, self).__init__()
        self.stochastic = stochastic
        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_in, 
                hidden_size=hidden_size, 
                num_layers=1, 
                bidirectional=False
            )
            if self.train_initial_hidden:
                self.h_0 = nn.Parameter(torch.randn(1, 1, hidden_size))
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_in, 
                hidden_size=hidden_size, 
                num_layers=1, 
                bidirectional=False
            )
            if self.train_initial_hidden:
                self.h_0 = nn.Parameter(torch.randn(1, 1, hidden_size))
                self.c_0 = nn.Parameter(torch.randn(1, 1, hidden_size))

        if self.stochastic:
            self.fc_mu = nn.Linear(hidden_size, dim_out)
            self.fc_logvar = nn.Linear(hidden_size, dim_out)
        else:
            self.fc = torch.nn.Linear(hidden_size, dim_out)
            
    def forward(self, x, h=None):
        l, n = x.shape[0], x.shape[1]
        if h is None:
            if self.train_initial_hidden:
                if self.net_type == "gru":
                    h_0 = self.h_0.repeat(1, n, 1)
                elif self.net_type == "lstm":
                    h_0 = (
                        self.h_0.repeat(1, n, 1),
                        self.c_0.repeat(1, n, 1)
                    )
                h_t, h_n = self.rnn(x, h_0)
            else:
                h_t, h_n = self.rnn(x)
        else:
            h_t, h_n = self.rnn(x, h)

        if self.stochastic:
            mu = self.fc_mu(h_t.reshape(-1, *h_t.shape[2:]))
            logvar = self.fc_logvar(h_t.reshape(-1, *h_t.shape[2:]))
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std
            z = z.reshape(l, n, *z.shape[1:])
            mu = mu.reshape(l, n, *mu.shape[1:])
            logvar = logvar.reshape(l, n, *logvar.shape[1:])
            return z, mu, logvar, h_n
        else:
            out = self.fc(h_t.reshape(-1, *h_t.shape[2:]))
            out = out.reshape(l, n, *out.shape[1:])
            return out, h_n