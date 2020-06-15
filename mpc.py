"""
MPC based on stochastic gradient descent.
"""
import torch
from torch import optim

class GradMPC():
    """
    This class solves the MPC problem based on SGD. Code inspired by: 
    https://github.com/homangab/gradcem/blob/master/mpc/grad.py.
    """
    def __init__(self, planning_horizon, opt_iters, env, device, grad_clip=True):
        self.set_env(env)
        self.device = device
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.grad_clip = grad_clip

    def to(self, device):
        self.device = device
        self.env.to(device=device)

    def set_env(self, env):
        self.env = env
        self.a_dim = env.action_space.shape
        self.x_dim = env.observation_space.shape

    def solve(self, initial_guess = None):
        """
        Args: 
            initial_guess: the initial guess for the solver (np.array or torch.tensor)
        Returns:
            actions: the final solution (torch.tensor)
        """
        if initial_guess == None:
            actions = torch.zeros((self.H, self.a_dim), device=self.device, requires_grad=True)
        else:
            assert isinstance(initial_guess, (np.ndarray, torch.Tensor))
            if type(initial_guess) is np.ndarray:
                initial_guess = torch.tensor(initial_guess)
            actions = initial_guess.clone().detach().to(device=self.device)
            actions.requires_grad = True

        #TODO: Perturb guess randomly or perturb and run multiple solves and take best result?

        opt = optim.SGD([actions], lr=0.1, momentum=0)

        #XXX: Every opt_iter is like an iteration of iterative MPC?
        for _ in range(self.opt_iters):
            cost = self.env.rollout(actions=actions)
            #TODO: Grad clip?
            opt.zero_grad()
            (-cost).backward()
            opt.step()

        return actions.detach()