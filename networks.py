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
        if bn: self.layers.append(nn.BatchNorm2d(128))
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
            if bn: self.layers.append(nn.BatchNorm2d(input, track_running_stats=True))
            if drop: self.layers.append(nn.Dropout(p=0.5))
        elif img_dim == 128:
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


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) # [450, 450, 100] --> 3
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            print("Level {}, Dilation {}, in channels {}, out channels{}".format(i, dilation_size, in_channels, out_channels))
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
