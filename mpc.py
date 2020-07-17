"""
MPC based on stochastic gradient descent.
"""
import torch
from torch import optim

class LinearMix(ModelWrapperMPC):
    def __init__(self, dyn_model, device="cpu"):
        self.dyn_model = dyn_model
        self.device = device
    
    def to(self, device):
        self.device = device
        self.dyn_model.to(device=device)

    def rollout(self, z_0, mu_0, var_0, a):
        """
        Args:
            z_0: Initial state sample (batch_size, dim_z)
            mu_0: Initial state mean (batch_size, dim_z)
            var_0: Initial distribution variance (batch_size, dim_z, dim_z)
            a: actions or controls (pred_len, batch_size, dim_a)
        Returns:
            z_hat: Predicted future states (pred_len, batch_size, dim_z)
            mu_hat: Predicted future means (pred_len, batch_size, dim_z)
            var_hat: Predicted future variances (pred_len, batch_size, dim_z, dim_z)
            info: {
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
                nets["dyn"](
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

        return z_hat, mu_hat, var_hat, info

class LinearMPC():
    """
    This class solves the locally linear MPC problem using off-the-shelf
    CVXOPT optimizers.
    """
    def __init__(self, planning_horizon, opt_iters, model):
        pass

class CEMMPC():
    """
    This class solves the MPC problem based on CEM. Code inspired by: 
    https://github.com/homangab/gradcem/blob/master/mpc/cem.py.
    """
    def __init__(self, planning_horizon, opt_iters, model, 
        device, samples=128, top_samples=16):
        pass

class GradMPC():
    """
    This class solves the MPC problem based on SGD. Code inspired by: 
    https://github.com/homangab/gradcem/blob/master/mpc/grad.py.
    """
    def __init__(self, planning_horizon, opt_iters, model, device, grad_clip=True):
        self.set_model(model)
        self.device = device
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.grad_clip = grad_clip

    def to(self, device):
        self.device = device
        self.model.to(device=device)

    def set_model(self, model):
        self.model = model
        self.a_dim = model.action_space.shape
        self.x_dim = model.observation_space.shape

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
            cost = self.model.rollout(actions=actions)
            #TODO: Grad clip?
            opt.zero_grad()
            (-cost).backward()
            opt.step()

        return actions.detach()