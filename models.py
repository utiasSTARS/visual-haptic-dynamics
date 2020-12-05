import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from networks import FullyConvEncoderVAE, FullyConvDecoderVAE

class LinearMixSSM(nn.Module):
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
    def __init__(
        self, dim_z, dim_u, hidden_size=128, 
        K=1, layers=1, bidirectional=False, 
        net_type="lstm", learn_uncertainty=False,
        train_initial_hidden=False
    ):
        super(LinearMixSSM, self).__init__()
        self.K = K
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.A = nn.Parameter(torch.eye(dim_z).repeat(K, 1, 1))
        self.B = nn.Parameter(torch.rand((K, dim_z, dim_u)))
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.train_initial_hidden = train_initial_hidden
        self.net_type = net_type

        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
            if self.train_initial_hidden:
                if bidirectional:
                    self.h_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                else:
                    self.h_0 = nn.Parameter(torch.randn(layers, 1, hidden_size))
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
            if self.train_initial_hidden:
                if bidirectional:
                    self.h_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                    self.c_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                else:
                    self.h_0 = nn.Parameter(torch.randn(layers * 1, 1, hidden_size))
                    self.c_0 = nn.Parameter(torch.randn(layers * 1, 1, hidden_size))

        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=K)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=K)

        self.softmax = nn.Softmax(dim=-1)
        
        self.learn_uncertainty = learn_uncertainty
        if self.learn_uncertainty: 
            self.fc_logvar = nn.Linear(hidden_size, dim_z)

    def forward(
        self, z_t, mu_t, var_t, u, 
        h_0=None, single=False, 
        return_matrices=False, return_all_hidden=False
    ):
        """
        Forward call to produce the subsequent state.

        Args:
            z_t: sampled state (seq_len, batch_size, dim_z)
            mu_t: state input mean (seq_len, batch_size, dim_z)
            var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
            u: control input (seq_len, batch_size, dim_u)
            h_0: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
               If None, h_0 is defaulted as 0-tensor
            single: If True then remove the need for a placeholder unsqueezed dimension for seq_len 
            return_matrices: Return state space matrices
            return_all_hidden: Return all hidden states h_t instead of only h_n
        Returns:
            z_t1: next sampled stats (seq_len, batch_size, dim_z)
            mu_t1: next state input mean (seq_len, batch_size, dim_z)
            var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
            h: hidden state(s) of the LSTM
        """
        if single:
            z_t = z_t.unsqueeze(0)
            u = u.unsqueeze(0)
            mu_t = mu_t.unsqueeze(0)
            var_t = var_t.unsqueeze(0)

        l, n, _ = z_t.shape
        inp = torch.cat([z_t, u], dim=-1)
        if h_0 is None:
            if self.train_initial_hidden:
                if self.net_type == "gru":
                    h_0 = self.h_0.repeat(1, n, 1)
                elif self.net_type == "lstm":
                    h_0 = (
                        self.h_0.repeat(1, n, 1),
                        self.c_0.repeat(1, n, 1)
                    )
                h_t, h_n = self.rnn(inp, h_0)
            else:
                h_t, h_n = self.rnn(inp)
        else:
            h_t, h_n = self.rnn(inp, h_0)

        if self.bidirectional:
            alpha = self.softmax(
                self.linear(h_t.reshape(-1, 2*self.hidden_size))
            ) # (seq_len * batch_size, k)
        else:
            alpha = self.softmax(
                self.linear(h_t.reshape(-1, self.hidden_size))
            ) # (seq_len * batch_size, k)

        z_t = z_t.reshape(-1, *z_t.shape[2:])
        mu_t = mu_t.reshape(-1, *mu_t.shape[2:])
        var_t = var_t.reshape(-1, *var_t.shape[2:])
        u = u.reshape(-1, *u.shape[2:])

        # Mixture of A
        A_t = torch.mm(alpha, self.A.reshape(-1, self.dim_z * self.dim_z)) # (l*bs, k) x (k, dim_z*dim_z) 
        A_t = A_t.reshape(-1, self.dim_z, self.dim_z) # (l*bs, dim_z, dim_z)

        # Mixture of B
        B_t = torch.mm(alpha, self.B.reshape(-1, self.dim_z * self.dim_u)) # (l*bs, k) x (k, dim_z*dim_z) 
        B_t = B_t.reshape(-1, self.dim_z, self.dim_u) # (l*bs, dim_z, dim_u)

        # Transition sample
        z_t1 = torch.bmm(A_t, z_t.unsqueeze(-1)) + torch.bmm(B_t, u.unsqueeze(-1))
        z_t1 = z_t1.reshape(l, n, *z_t1.shape[1:]).squeeze(-1)

        # Transition distribution
        mu_t1 = z_t1

        if self.learn_uncertainty:
            if self.bidirectional:
                logvar_t1 = self.fc_logvar(h_t.reshape(-1, 2*self.hidden_size)) # (seq_len * batch_size, dim_z)
            else:
                logvar_t1 = self.fc_logvar(h_t.reshape(-1, self.hidden_size)) # (seq_len * batch_size, dim_z)
            var_t1 = torch.diag_embed(torch.exp(logvar_t1)) # (seq_len * batch_size, dim_z, dim_z)
            var_t1 = var_t1.reshape(l, n, *var_t1.shape[1:])
        else:
            Q = torch.eye(self.dim_z, requires_grad=False, device=z_t.device) 
            var_t1 = 0.01 * Q.repeat(l, n, 1, 1)

        A_t = A_t.reshape(l, n, *A_t.shape[1:])
        B_t = B_t.reshape(l, n, *B_t.shape[1:])

        if single:
            z_t1 = z_t1[0]
            mu_t1 = mu_t1[0]
            var_t1 = var_t1[0]
            A_t = A_t[0]
            B_t = B_t[0]

        if return_all_hidden:
            h = (h_t, h_n)
        else:
            h = h_n

        if return_matrices:
            return z_t1, mu_t1, var_t1, h, A_t, B_t

        return z_t1, mu_t1, var_t1, h

class NonLinearSSM(nn.Module):
    """
    This class defines the GRU-based or LSTM-based non-linear
    dynamics network inspired by https://arxiv.org/abs/1506.07365. 

    Args:
        input_size: Input dimension
        dim_z: Dimension of state
        dim_u: Dimension of action
        hidden_size: Hidden state dimension
        layers: Number of layers
        bidirectional: Use bidirectional version
        net_type: Use the LSTM or GRU variation
    """
    def __init__(
        self, dim_z, dim_u, hidden_size=128,
        layers=1, bidirectional=False, 
        net_type="lstm", train_initial_hidden=False
    ):    
        super(NonLinearSSM, self).__init__()
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.train_initial_hidden = train_initial_hidden
        self.net_type = net_type

        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
            if self.train_initial_hidden:
                if bidirectional:
                    self.h_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                else:
                    self.h_0 = nn.Parameter(torch.randn(layers, 1, hidden_size))
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
            if self.train_initial_hidden:
                if bidirectional:
                    self.h_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                    self.c_0 = nn.Parameter(torch.randn(layers * 2, 1, hidden_size))
                else:
                    self.h_0 = nn.Parameter(torch.randn(layers * 1, 1, hidden_size))
                    self.c_0 = nn.Parameter(torch.randn(layers * 1, 1, hidden_size))

        if bidirectional:
            self.fc_mu = nn.Linear(2*hidden_size, dim_z)
            self.fc_logvar = nn.Linear(2*hidden_size, dim_z)
        else:
            self.fc_mu = nn.Linear(hidden_size, dim_z)
            self.fc_logvar = nn.Linear(hidden_size, dim_z)
        
    def forward(self, z_t, mu_t, var_t, u, h_0=None, single=False, return_all_hidden=False):
        """
        Forward call to produce the subsequent state.

        Args:
            z_t: sampled state (seq_len, batch_size, dim_z)
            mu_t: state input mean (seq_len, batch_size, dim_z)
            var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
            u: control input (seq_len, batch_size, dim_u)
            h_0: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
            If None, h is defaulted as 0-tensor
            single: If True then remove the need for a placeholder unsqueezed dimension for seq_len
            return_all_hidden: Return all hidden states h_t instead of only h_n
        Returns:
            z_t1: next sampled stats (seq_len, batch_size, dim_z)
            mu_t1: next state input mean (seq_len, batch_size, dim_z)
            var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
            h: hidden state(s) of the LSTM
        """
        if single:
            z_t = z_t.unsqueeze(0)
            u = u.unsqueeze(0)
            mu_t = mu_t.unsqueeze(0)
            var_t = var_t.unsqueeze(0)

        l, n, _ = z_t.shape

        inp = torch.cat([z_t, u], dim=-1)
        if h_0 is None:
            if self.train_initial_hidden:
                if self.net_type == "gru":
                    h_0 = self.h_0.repeat(1, n, 1)
                elif self.net_type == "lstm":
                    h_0 = (
                        self.h_0.repeat(1, n, 1),
                        self.c_0.repeat(1, n, 1)
                    )
                h_t, h_n = self.rnn(inp, h_0)
            else:
                h_t, h_n = self.rnn(inp)
        else:
            h_t, h_n = self.rnn(inp, h_0)

        if self.bidirectional:
            mu_t1 = self.fc_mu(h_t.reshape(-1, 2*self.hidden_size)) # (seq_len * batch_size, dim_z)
            logvar_t1 = self.fc_logvar(h_t.reshape(-1, 2*self.hidden_size)) # (seq_len * batch_size, dim_z)
        else:
            mu_t1 = self.fc_mu(h_t.reshape(-1, self.hidden_size)) # (seq_len * batch_size, dim_z)
            logvar_t1 = self.fc_logvar(h_t.reshape(-1, self.hidden_size)) # (seq_len * batch_size, dim_z)

        var_t1 = torch.diag_embed(torch.exp(logvar_t1)) # (seq_len * batch_size, dim_z, dim_z)

        # Reparameterized sample
        std_t1 = torch.exp(logvar_t1 / 2.0)
        eps = torch.randn_like(std_t1)
        z_t1 = mu_t1 + eps * std_t1

        z_t1 = z_t1.reshape(l, n, *z_t1.shape[1:])
        mu_t1 = mu_t1.reshape(l, n, *mu_t1.shape[1:])
        var_t1 = var_t1.reshape(l, n, *var_t1.shape[1:])

        if single:
            z_t1 = z_t1[0]
            mu_t1 = mu_t1[0]
            var_t1 = var_t1[0]
            
        if return_all_hidden:
            h = (h_t, h_n)
        else:
            h = h_n

        return z_t1, mu_t1, var_t1, h

class ProductOfExperts(nn.Module):
    """A generalized product of M experts.
    Implementation based on https://github.com/mhw32/multimodal-vae-public/.
    mu: (bs x M x D)
    logvar: (bs x M X D)
    """
    def forward(self, mu, logvar, eps=1e-8, prior=True):
        if prior:
            bs, d = mu.shape[0], mu.shape[2]
            device = mu.device
            mu = torch.cat((
                mu, 
                torch.zeros((bs, 1, d), requires_grad=False, device=device)
            ), axis=1)
            logvar = torch.cat((
                logvar, 
                torch.zeros((bs, 1, d), requires_grad=False, device=device)
            ), axis=1)

        var = torch.exp(logvar) + eps
        # Precision
        T = 1.0 / (var + eps)
        mu_pd = torch.sum(mu * T, dim=1) / torch.sum(T, dim=1)
        var_pd = 1.0 / torch.sum(T, dim=1)
        logvar_pd = torch.log(var_pd + eps)
        return mu_pd, logvar_pd
