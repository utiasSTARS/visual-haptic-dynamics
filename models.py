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
        net_type="lstm"
    ):
        super(LinearMixSSM, self).__init__()
        self.K = K
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.A = nn.Parameter(torch.eye(dim_z).repeat(K, 1, 1))
        self.B = nn.Parameter(torch.rand((K, dim_z, dim_u)))
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=K)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=K)

        self.softmax = nn.Softmax(dim=-1)
    
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
            h_t, h_n = self.rnn(inp)
        else:
            if single:
                h_0 = h_0.unsqueeze(0)
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
            h_t = h_t[0]
            h_n = h_n[0]
        
        if return_all_hidden:
            h = (h_t, h_n)
        else:
            h = h_n

        if return_matrices:
            return z_t1, mu_t1, var_t1, h, A_t, B_t

        return z_t1, mu_t1, var_t1, h


class LinearSSM(nn.Module):
    """
    This class defines the GRU-based or LSTM-based rank-1 approximation 
    linear dynamics network inspired by https://arxiv.org/abs/1506.07365. 

    Args:
        input_size: Input dimension
        dim_z: Dimension of state
        dim_u: Dimension of action
        hidden_size: Hidden state dimension
        layers: Number of layers
        bidirectional: Use bidirectional version
        net_type: Use the LSTM or GRU variation
    """
    def __init__(self, dim_z, dim_u, hidden_size=128,
                layers=1, bidirectional=False, net_type="lstm"):
        super(LinearSSM, self).__init__()
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        n_outputs = 2 * dim_z + dim_z * dim_u + dim_z 

        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=n_outputs)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=n_outputs)

        self.sigmoid = nn.Sigmoid()

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
            h_t, h_n = self.rnn(inp)
        else:
            h_t, h_n = self.rnn(inp, h_0)

        if self.bidirectional:
            out = self.sigmoid(
                self.linear(h_t.reshape(-1, 2*self.hidden_size))
            )
        else:
            out = self.sigmoid(
                self.linear(h_t.reshape(-1, self.hidden_size))
            )

        I = torch.eye(self.dim_z, requires_grad=False, device=z_t.device) 

        r = out[:, 0:self.dim_z]
        v = out[:, self.dim_z:(2 * self.dim_z)]
        A_t = I + torch.bmm(r.unsqueeze(2), v.unsqueeze(1))
        B_t = out[:, (2 * self.dim_z):(2 * self.dim_z + self.dim_z * self.dim_u)]
        B_t = B_t.view(-1, self.dim_z, self.dim_u) # reshape into matrix
        o_t = out[:, (2 * self.dim_z + self.dim_z * self.dim_u):]

        z_t = z_t.reshape(-1, *z_t.shape[2:])
        mu_t = mu_t.reshape(-1, *mu_t.shape[2:])
        var_t = var_t.reshape(-1, *var_t.shape[2:])
        u = u.reshape(-1, *u.shape[2:])

        # Transition sample
        z_t1 = torch.bmm(A_t, z_t.unsqueeze(-1)) \
            + torch.bmm(B_t, u.unsqueeze(-1)) \
            + o_t.unsqueeze(-1)
        z_t1 = z_t1.reshape(l, n, *z_t1.shape[1:]).squeeze(-1)

        # Transition mean
        mu_t1 = torch.bmm(A_t, mu_t.unsqueeze(-1)) \
            + torch.bmm(B_t, u.unsqueeze(-1)) \
            + o_t.unsqueeze(-1)
        mu_t1 = mu_t1.reshape(l, n, *mu_t1.shape[1:]).squeeze(-1)

        # Transition covariance
        var_t1 = torch.bmm(torch.bmm(A_t, var_t), A_t.transpose(1, 2)) + I
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
    def __init__(self, dim_z, dim_u, hidden_size=128,
                layers=1, bidirectional=False, net_type="lstm"):    
        super(NonLinearSSM, self).__init__()
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if net_type == "gru":
            self.rnn = nn.GRU(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z + dim_u, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
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
            h_t, h_n = self.rnn(inp)
        else:
            if single:
                h_0 = h_0.unsqueeze(0)
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
            h_t = h_t[0]
            h_n = h_n[0]
            
        if return_all_hidden:
            h = (h_t, h_n)
        else:
            h = h_n

        return z_t1, mu_t1, var_t1, h