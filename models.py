import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from networks import FullyConvEncoderVAE

class ActorCriticCNN(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, device, shared_hidden_dim=256, img_dim=(1,64,64)):
        super(ActorCriticCNN, self).__init__()

        # shared network
        self.shared_net = FullyConvEncoderVAE(
            input=img_dim[0], 
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

        self.action_var = nn.Parameter(torch.full((action_dim,), action_std*action_std).to(device))
        self.device = device

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        hidden = self.shared_net(state)
        action_mean = self.actor_head(hidden)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.detach()
    
    def evaluate(self, state, action): 
        hidden = self.shared_net(state)
        action_mean = self.actor_head(hidden)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic_head(hidden)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class ActorCriticMLP(nn.Module):
    def __init__(self, state_dim, action_dim, action_std, device):
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
        self.action_var = nn.Parameter(torch.full((action_dim,), action_std*action_std).to(device))
        self.device = device

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()
    
    def evaluate(self, state, action):  
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy