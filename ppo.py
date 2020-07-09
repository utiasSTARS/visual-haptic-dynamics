"""
Minimalistic PPO implementation 
from https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO_continuous.py#L128.
"""
import torch
import torch.nn as nn
import numpy as np
import copy
from utils import common_init_weights

class SAC:
    def __init__(self, lr, gamma, K_epochs, eps_clip, device, actor_critic):
        pass
        
class PPO:
    def __init__(self, lr, gamma, K_epochs, eps_clip, device, actor_critic, batch_size):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = actor_critic
        self.policy.apply(common_init_weights)
        
        self.MseLoss = nn.MSELoss()
        self.device = device
        self.batch_size = batch_size

        self.setup_opt(lr)

    def setup_opt(self, lr):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(self.device)
        return self.policy.act(state, memory).cpu().numpy().flatten()
    
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
        states = torch.squeeze(torch.stack(list(memory.states)), 1)
        actions = torch.squeeze(torch.stack(list(memory.actions)), 1)
        logprobs = torch.squeeze(torch.stack(list(memory.logprobs)), 1)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            idx = np.arange(returns.shape[0])
            np.random.shuffle(idx)
            for ii in range(idx.shape[0] // self.batch_size):
                opt_idx = idx[ii * self.batch_size:(ii + 1) * self.batch_size]
                self.opt_step(
                    states[opt_idx], 
                    actions[opt_idx], 
                    logprobs[opt_idx], 
                    returns[opt_idx]
                )

    def opt_step(self, states, actions, initial_logprobs, returns):
        # Evaluating old actions and values :
        logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
        
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - initial_logprobs)

        # Finding Surrogate Loss:
        advantages = returns - state_values.detach() # A = Q - V
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - 0.01 * dist_entropy

        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()