"""
MPC based on stochastic gradient descent.
"""
import torch
from torch import optim

class LinearMixWrapper():
    def __init__(self, dyn_model, device="cpu"):
        self.dyn_model = dyn_model
        self.device = device
    
    def to(self, device):
        self.device = device
        self.dyn_model.to(device=device)

    def rollout(self, s_0, a):
        """
        Args:
            s_0: Initial state (z_0, mu_0, var_0)
                z_0: Initial state sample (batch_size, dim_z)
                mu_0: Initial state mean (batch_size, dim_z)
                var_0: Initial distribution variance (batch_size, dim_z, dim_z)
            a: actions or controls (pred_len, batch_size, dim_a)
        Returns:
            z_hat: Predicted future states (pred_len, batch_size, dim_z)
            info: Any extra information needed {
                A: Locally linear transition matrices (pred_len, batch_size, dim_z, dim_z)
                B: Locally linear control matrices (pred_len, batch_size, dim_z, dim_a)
            }
        """
        pred_len, batch_size = a.shape[0], a.shape[1]
        dim_z, dim_a = z0.shape[-1], a.shape[-1]

        z_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        mu_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        var_hat = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        A = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        B = torch.zeros((pred_len, batch_size, dim_z, dim_a)).float().to(device=self.device)

        h_0 = None
        for ll in pred_len:
            z_t1_hat, mu_z_t1_hat, var_z_t1_hat, h_t1, A_t1, B_t1 = \
                self.dyn_model(
                    z_t=z_0, 
                    mu_t=mu_0, 
                    var_t=var_0, 
                    u=u[ll],
                    h=h_0,
                    single=True,
                    return_matrices=True
                )
            
            z_hat[ll] = z_t1_hat
            mu_hat[ll] = mu_z_t1_hat
            var_hat[ll] = var_z_t1_hat
            A[ll] = A_t1
            B[ll] = B_t1
            z_0, mu_0, var_0, h_0 = z_t1_hat, mu_z_t1_hat, var_z_t1_hat, h_t1    

        info = {
            "A": A,
            "B": B
        }

        return z_hat, info

class MPC():
    """
    Generic MPC solver based on PyTorch transition or dynamics.
    """
    def __init__(self, planning_horizon, opt_iters, model, device):
        self.set_model(model)
        self.device = device
        self.H = planning_horizon
        self.opt_iters = opt_iters

    def to(self, device):
        self.device = device
        self.model.to(device=device)

    def set_model(self, model):
        self.model = model

    def solve(self, initial_state, initial_guess=None):
        raise NotImplementedError()

class CVXLinear(MPC):
    """
    This class solves the locally linear MPC problem using off-the-shelf
    CVXOPT optimizers.
    """
    def __init__(
        self, 
        planning_horizon, 
        opt_iters, 
        model, 
        goal, 
        device):

        super(CVXLinear, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = goal

    def cost(self, s_hat, info):
        pass

    def solve(self, initial_state, initial_guess=None):
        pass

class CEM(MPC):
    """
    This class solves the MPC problem based on CEM. Code inspired by: 
    https://github.com/homangab/gradcem/blob/master/mpc/cem.py.
    """
    def __init__(
        self, 
        planning_horizon, 
        opt_iters, 
        model, 
        goal, 
        device, 
        samples=128, 
        top_samples=16):

        super(CEM, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )
        self.samples = samples
        self.top_samples = top_samples
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = goal

    def cost(self, s_hat, info):
        pass

    def solve(self, initial_state, initial_guess=None):
        pass

class Grad(MPC):
    """
    This class solves the MPC problem based on SGD. Code inspired by: 
    https://github.com/homangab/gradcem/blob/master/mpc/grad.py.
    """
    def __init__(
        self, 
        planning_horizon, 
        opt_iters, 
        model, 
        goal, 
        device, 
        grad_clip=True):

        super(Grad, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )
        self.grad_clip = grad_clip
        self.set_goal(goal)

    def set_goal(self, goal):
        self.goal = goal

    def cost(self, s_hat, info):
        return self.goal - s_hat
        
    def solve(self, initial_state, initial_guess=None):
        """
        Args: 
            initial_guess: the initial guess for the solver (np.array or torch.tensor)
        Returns:
            actions: the final solution (torch.tensor)
        """
        if initial_guess == None:
            actions = torch.zeros(
                (self.H, self.a_dim), 
                device=self.device, 
                requires_grad=True
            )
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
            s_hat, info = self.model.rollout(initial_state=initial_state, actions=actions)
            cost = self.cost(s_hat, info)
            #TODO: Grad clip?
            opt.zero_grad()
            (-cost).backward()
            opt.step()

        return actions.detach()