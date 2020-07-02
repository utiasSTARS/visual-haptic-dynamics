"""
Minimalistic PPO implementation 
from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py#L128.
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from utils import common_init_weights

class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip, device, actor_critic):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = actor_critic
        self.policy.apply(common_init_weights)
        
        self.MseLoss = nn.MSELoss()
        self.device = device

        self.setup_opt(lr)

    def setup_opt(self, lr):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def opt_step(self, old_states, old_actions, old_logprobs, returns):
        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs)

        # Finding Surrogate Loss:
        advantages = returns - state_values.detach()   
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) - 0.01*dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(self.device)
        return self.policy.act(state, memory).cpu().data.numpy().flatten()
    
    def update(self, memory):
        # Monte Carlo estimate of rewards:
        returns = []
        discounted_return = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_return = 0
            discounted_return = reward + (self.gamma * discounted_return)
            returns.insert(0, discounted_return)
        
        # Normalizing the returns:
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        returns = returns.float()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(self.device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(self.device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            self.opt_step(old_states, old_actions, old_logprobs, returns)

class AuxPPO(PPO):
    def __init__(self, lr, gamma, K_epochs, eps_clip, device, actor_critic):
        super(AuxPPO, self).__init__(lr, gamma, K_epochs, eps_clip, device, actor_critic)

    def setup_opt(self, lr):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def opt_step(self, old_states, old_actions, old_logprobs, returns):
        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy, state_hat = self.policy.evaluate(old_states, old_actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs)

        # Finding Surrogate Loss:
        advantages = returns - state_values.detach()   
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        rec = self.MseLoss(state_hat, old_states)
        loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, returns) + rec - 0.01*dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()