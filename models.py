import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from networks import FullyConvEncoderVAE, FullyConvDecoderVAE, TemporalConvNet

class ActorCriticMLP(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCriticMLP, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )

        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = nn.Parameter(torch.full((action_dim,), action_std*action_std))
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state.detach().clone())
        memory.actions.append(action.detach().clone())
        memory.logprobs.append(action_logprob.detach().clone())
        return action.detach().clone()
    
    def evaluate(self, state, action):  
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class ActorCriticCNN(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, shared_hidden_dim=256, img_dim=(64,64,1)):
        super(ActorCriticCNN, self).__init__()

        # shared network
        self.shared_net = FullyConvEncoderVAE(
            input=img_dim[2], 
            latent_size=shared_hidden_dim, 
            bn=False, 
            drop=False, 
            nl=nn.Tanh(), 
            stochastic=False, 
            img_dim=img_dim[1]
        )

        # actor head, action mean range -1 to 1
        self.actor_head =  nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # critic head
        self.critic_head = nn.Sequential(
            nn.Linear(shared_hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.action_var = nn.Parameter(torch.full((action_dim,), action_std*action_std))

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        hidden = self.shared_net(state)
        action_mean = self.actor_head(hidden)
        cov_mat = torch.diag(self.action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state.detach())
        memory.actions.append(action.detach())
        memory.logprobs.append(action_logprob.detach())
        return action.detach()
    
    def evaluate(self, state, action): 
        hidden = self.shared_net(state)
        action_mean = self.actor_head(hidden.detach())

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_head(hidden)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

        
class AuxActorCriticCNN(ActorCriticCNN):
    def __init__(self, state_dim, action_dim, action_std, shared_hidden_dim=256, img_dim=(64,64,1)):
        super(AuxActorCriticCNN, self).__init__(
            state_dim, 
            action_dim, 
            action_std, 
            shared_hidden_dim=shared_hidden_dim, 
            img_dim=img_dim
        )

        self.decoder = FullyConvDecoderVAE(
            input=img_dim[2],
            latent_size=shared_hidden_dim,
            bn=False,
            drop=False,
            nl=nn.Tanh(),
            img_dim=img_dim[1],
            output_nl=nn.Sigmoid()
        )

    def evaluate(self, state, action): 
        hidden = self.shared_net(state)
        action_mean = self.actor_head(hidden.detach())

        state_hat = self.decoder(hidden)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_head(hidden)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy, state_hat


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
    def __init__(self, dim_z, dim_u, hidden_size=128, 
                 K=1, layers=1, bidirectional=False, net_type="lstm"):
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
                input_size=dim_z, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=K)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=K)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, z_t, mu_t, var_t, u, h=None, single=False):
        """
        Forward call to produce the subsequent state.

        Args:
            z_t: sampled state (seq_len, batch_size, dim_z)
            mu_t: state input mean (seq_len, batch_size, dim_z)
            var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
            u: control input (seq_len, batch_size, dim_u)
            h: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
               If None, h is defaulted as 0-tensor
            single: If True then remove the need for a placeholder unsqueezed dimension for seq_len 
        Returns:
            z_t1: next sampled stats (seq_len, batch_size, dim_z)
            mu_t1: next state input mean (seq_len, batch_size, dim_z)
            var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
            h: hidden state of the LSTM
        """
        if single:
            z_t = z_t.unsqueeze(0)
            u = u.unsqueeze(0)
            mu_t = mu_t.unsqueeze(0)
            var_t = var_t.unsqueeze(0)

        n, l, _ = z_t.shape

        if h is None:
            x, h = self.rnn(z_t)
        else:
            x, h = self.rnn(z_t, h)
        
        if self.bidirectional:
            x = x.reshape(-1, 2*self.hidden_size) # (seq_len * batch_size, 2 * hidden_size)
        else:
            x = x.reshape(-1, self.hidden_size) # (seq_len * batch_size, hidden_size)

        alpha = self.softmax(self.linear(x)) # (seq_len * batch_size, k)

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
        z_t1 = z_t1.reshape(n, l, *z_t1.shape[1:]).squeeze(-1)

        # Transition distribution
        mu_t1 = z_t1
        Q = torch.eye(self.dim_z, requires_grad=False, device=z_t.device) 
        var_t1 = 0.01 * Q.repeat(n, l, 1, 1)

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
                input_size=dim_z, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        if bidirectional:
            self.linear = nn.Linear(in_features=2*hidden_size, out_features=n_outputs)
        else:
            self.linear = nn.Linear(in_features=hidden_size, out_features=n_outputs)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z_t, mu_t, var_t, u, h=None, single=False):
        """
        Forward call to produce the subsequent state.

        Args:
            z_t: sampled state (seq_len, batch_size, dim_z)
            mu_t: state input mean (seq_len, batch_size, dim_z)
            var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
            u: control input (seq_len, batch_size, dim_u)
            h: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
               If None, h is defaulted as 0-tensor
            single: If True then remove the need for a placeholder unsqueezed dimension for seq_len
        Returns:
            z_t1: next sampled stats (seq_len, batch_size, dim_z)
            mu_t1: next state input mean (seq_len, batch_size, dim_z)
            var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
            h: hidden state of the LSTM
        """
        if single:
            z_t = z_t.unsqueeze(0)
            u = u.unsqueeze(0)
            mu_t = mu_t.unsqueeze(0)
            var_t = var_t.unsqueeze(0)

        l, n, _ = z_t.shape

        if h is None:
            x, h = self.rnn(z_t)
        else:
            x, h = self.rnn(z_t, h)
        
        if self.bidirectional:
            x = x.reshape(-1, 2*self.hidden_size) # (seq_len * batch_size, 2 * hidden_size)
        else:
            x = x.reshape(-1, self.hidden_size) # (seq_len * batch_size, hidden_size)

        out = self.sigmoid(self.linear(x)) # (seq_len * batch_size, n_outputs)

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
                input_size=dim_z, 
                hidden_size=hidden_size, 
                num_layers=layers, 
                bidirectional=bidirectional
            )
        elif net_type =="lstm":
            self.rnn = nn.LSTM(
                input_size=dim_z, 
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

    def forward(self, z_t, mu_t, var_t, u, h=None, single=False):
            """
            Forward call to produce the subsequent state.

            Args:
                z_t: sampled state (seq_len, batch_size, dim_z)
                mu_t: state input mean (seq_len, batch_size, dim_z)
                var_t: state input covariance (seq_len, batch_size, dim_z, dim_z)
                u: control input (seq_len, batch_size, dim_u)
                h: hidden state of the LSTM (num_layers * num_directions, batch_size, hidden_size) or None. 
                If None, h is defaulted as 0-tensor
                single: If True then remove the need for a placeholder unsqueezed dimension for seq_len
            Returns:
                z_t1: next sampled stats (seq_len, batch_size, dim_z)
                mu_t1: next state input mean (seq_len, batch_size, dim_z)
                var_t1: next state input covariance (seq_len, batch_size, dim_z, dim_z)
                h: hidden state of the LSTM
            """
            if single:
                z_t = z_t.unsqueeze(0)
                u = u.unsqueeze(0)
                mu_t = mu_t.unsqueeze(0)
                var_t = var_t.unsqueeze(0)

            l, n, _ = z_t.shape

            if h is None:
                x, h = self.rnn(z_t)
            else:
                x, h = self.rnn(z_t, h)
            
            if self.bidirectional:
                x = x.reshape(-1, 2*self.hidden_size) # (seq_len * batch_size, 2 * hidden_size)
            else:
                x = x.reshape(-1, self.hidden_size) # (seq_len * batch_size, hidden_size)

            mu_t1 = self.fc_mu(x) # (seq_len * batch_size, dim_z)
            logvar_t1 = self.fc_logvar(x) # (seq_len * batch_size, dim_z)
            var_t1 = torch.diag_embed(torch.exp(logvar_t1)) # (seq_len * batch_size, dim_z, dim_z)

            # Reparameterized sample
            std_t1 = torch.exp(logvar_t1 / 2.0)
            eps = torch.randn_like(std_t1)
            z_t1 = mu_t1 + eps * std_t1

            z_t1 = z_t1.reshape(l, n, *z_t1.shape[1:])
            mu_t1 = mu_t1.reshape(l, n, *mu_t1.shape[1:])
            var_t1 = var_t1.reshape(l, n, *var_t1.shape[1:])

            return z_t1, mu_t1, var_t1, h


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout) # num_channels = [450, 450, 100]

    def forward(self, x):
        y = self.tcn(x.transpose(1, 2))
        return y.transpose(1, 2)