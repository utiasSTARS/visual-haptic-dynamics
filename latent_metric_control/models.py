"""
Learned models to test metrics with.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
import time


class FullyConvEncoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, bn=True, 
                 drop=False, nl=nn.ReLU(), stochastic=True, img_dim="64"):
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

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        if self.stochastic:
            self.fc_mu = nn.Linear(n_size, latent_size + extra_scalars_conc)
            self.fc_logvar = nn.Linear(n_size, latent_size)
        else:
            self.fc = nn.Linear(n_size, latent_size)

        self.flatten = Flatten()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = self.flatten(x)
        if self.stochastic:
            x_mu = self.fc_mu(x)
            mu = x_mu[:, :self.latent_size]
            logvar = self.fc_logvar(x)
            # Reparameterize
            std = torch.exp(logvar / 2.0)
            eps = torch.randn_like(std)
            z = mu + eps * std

            return z, mu, logvar
        else: 
            return x


class FullyConvDecoderVAE(nn.Module):
    def __init__(self, input=1, latent_size=12, output_nl=nn.Tanh(), bn=True, 
                    drop=False, nl=nn.ReLU(), img_dim="64"):
        super(FullyConvDecoderVAE, self).__init__()
        self.bn = bn
        self.drop = drop
        self.layers = nn.ModuleList()

        if img_dim == "64":
            n_size = 256 * 2 * 2
        elif img_dim == "128":
            n_size = 256 * 6 * 6
        else:
            raise NotImplementedError()

        self.layers.append(nn.ConvTranspose2d(n_size, 128, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(128))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        self.layers.append(nn.ConvTranspose2d(128, 64, 5, stride=2, bias=False))
        if bn: self.layers.append(nn.BatchNorm2d(64, track_running_stats=True))
        if drop: self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nl)

        if img_dim == "64":
            self.layers.append(nn.ConvTranspose2d(64, 32, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, input, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(input, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
        elif img_dim == "128":
            self.layers.append(nn.ConvTranspose2d(64, 32, 5, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(32, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
            self.layers.append(nl)

            self.layers.append(nn.ConvTranspose2d(32, 16, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(16, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))

            self.layers.append(nn.ConvTranspose2d(16, input, 6, stride=2, bias=False))
            if bn: self.layers.append(nn.BatchNorm2d(input, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
        else:
            raise NotImplementedError()

        if output_nl != None:
            self.layers.append(output_nl)

        self.linear = nn.Linear(latent_size, n_size, bias=False)
        self.batchn = nn.BatchNorm1d(n_size)
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
    def __init__(self, dim_in, dim_out, bn=False, drop=False, nl=nn.ReLU(), hidden_size=800, stochastic=True):
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
            print(x.shape)
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
        x = z.view(-1, *self.dim_out)
        return x


class LinearMixRNN(nn.Module):
    """
    This class defines the GRU-based or LSTM-based linear mixture dynamics network from
    https://github.com/simonkamronn/kvae/blob/master/kvae/filter.py.

    Args:
        input_size: Input dimension
        dim_z: Dimension of state
        dim_u: Dimension of action
        hidden_size: Hidden state dimension
        K: Mixture amount
        layers: Number of layers
        bidirectional: Use bidirectional version
        net_type: Use the LSTM or GRU variation
    """
    def __init__(self, input_size, dim_z, dim_u, hidden_size=128, 
                 K=1, layers=1, bidirectional=False, net_type="lstm"):
        super(LinearMixRNN, self).__init__()
        self.K = K
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.A = nn.Parameter(torch.eye(dim_z).repeat(K, 1, 1))
        self.B = nn.Parameter(torch.rand((K, dim_z, dim_u)))
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.layers = layers
        if net_type == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=layers, bidirectional=bidirectional)
        elif net_type =="lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                num_layers=layers, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=K)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=K)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, z_t, mu_t, var_t, u, h=None):
        """
        Forward call to produce the subsequent state.

        Args:
            z_t: sampled state (seq_len, batch_size, dim_z)
            mu_t: state input mean (seq_len, batch_size, dim_z)
            var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
            u: control input (seq_len, batch_size, dim_u)
            h: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
               If None, h is defaulted as 0-tensor
        Returns:
            z_t1: next sampled stats (seq_len, batch_size, dim_z)
            mu_t1: next state input mean (seq_len, batch_size, dim_z)
            var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
            h: hidden state of the LSTM
        """
        l, n, _ = z_t.shape
        if h is None:
            x, h = self.rnn(z_t)
        else:
            x, h = self.rnn(z_t, h)
        
        if self.bidirectional:
            x = x.reshape(l * n, 2*self.hidden_size) # (seq_len * batch_size, 2 * hidden_size)
        else:
            x = x.reshape(l * n, self.hidden_size) # (seq_len * batch_size, hidden_size)

        alpha = self.softmax(self.linear(x))
        alpha = alpha.reshape(l, n, self.K) # (seq_len, batch_size, K)

        # Mixture of A
        A_t = torch.mm(alpha_t.reshape(-1, self.K), self.A.reshape(-1, self.dim_z * self.dim_z)) # (l*bs, k) x (k, dim_z*dim_z) 
        A_t = A_t.reshape(-1, self.dim_z, self.dim_z) # (l*bs, dim_z, dim_z)

        # Mixture of B
        B_t = torch.mm(alpha_t.reshape(-1, self.K), self.B.reshape(-1, self.dim_z * self.dim_u)) # (l*bs, k) x (k, dim_z*dim_z) 
        B_t = B_t.reshape(-1, self.dim_z, self.dim_u) # (l*bs, dim_z, dim_u)

        # Transition sample
        z_t1 = torch.bmm(A_t, z.reshape(-1, self.dim_z).unsqueeze(-1)) + 
               torch.bmm(B_t, u.reshape(-1, self.dim_u).unsqueeze(-1))
        z_t1 = z_t1.squeeze(-1).reshape(l, n, -1)

        # Transition mean
        mu_t1 = torch.bmm(A_t, mu_t.reshape(-1, self.dim_z).unsqueeze(-1)) + 
                torch.bmm(B_t, u.reshape(-1, self.dim_u).unsqueeze(-1))
        mu_t1 = mu_t1.squeeze(-1).reshape(l, n, -1)

        # Transition covariance
        A_t_tr = torch.transpose(A_t, 0, 1)
        # Noise matrix
        I = torch.eye(self.dim_z, requires_grad=False, device=z_t.device) 
        var_t1 = torch.bmm(torch.bmm(A_t, var_t.reshape(-1, self.dim_z, self.dim_z)), A_t_tr) + I
        var_t1 = var_t1.reshape(l, n, self.dim_z, self.dim_z)
        return z_t1, mu_t1, var_t1, h


class LinearRNN(nn.module):
    pass

class NonLinearRNN(nn.module):
    #TODO: See NYU Deep SSM on how to transition covariance
    pass