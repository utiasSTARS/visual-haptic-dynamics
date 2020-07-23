"""
MPC based on stochastic gradient descent.
"""
import torch
from torch import optim
import numpy as np
import cvxpy as cp
import scipy.sparse as sparse

class LinearMixWrapper():
    def __init__(self, dyn_model, device="cpu"):
        self.dyn_model = dyn_model
        self.device = device
    
    def to(self, device):
        self.device = device
        self.dyn_model.to(device=device)

    def rollout(self, z_0, mu_0, var_0, u):
        """
        Args:
            z_0: Initial state sample (batch_size, dim_z)
            mu_0: Initial state mean (batch_size, dim_z)
            var_0: Initial distribution variance (batch_size, dim_z, dim_z)
            u: Controls (pred_len, batch_size, dim_u)
        Returns:
            mu_hat: Predicted future states (pred_len, batch_size, dim_z)
            info: Any extra information needed {
                A: Locally linear transition matrices (pred_len, batch_size, dim_z, dim_z)
                B: Locally linear control matrices (pred_len, batch_size, dim_z, dim_u)
            }
        """
        pred_len, batch_size = u.shape[0], u.shape[1]
        dim_z, dim_u = z_0.shape[-1], u.shape[-1]

        z_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        mu_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        var_hat = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        A = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        B = torch.zeros((pred_len, batch_size, dim_z, dim_u)).float().to(device=self.device)

        h_0 = None
        for ll in range(pred_len):
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

        #XXX: Track based on mean for now
        return mu_hat, info

class MPC():
    """
    Generic MPC solver based on PyTorch transition or dynamics.
    """
    def __init__(self, planning_horizon, opt_iters, model, device="cpu"):
        self.set_model(model)
        self.to(device)
        self.H = planning_horizon
        self.opt_iters = opt_iters

    def to(self, device):
        self.device = device
        self.model.to(device=device)

    def set_model(self, model):
        self.model = model

    def solve(self, z_0, mu_0, var_0, z_g, u_0=None):
        raise NotImplementedError()

class CVXLinear(MPC):
    """
    This class solves the locally linear MPC problem using off-the-shelf
    cvxpy optimizers.
    """
    def __init__(
        self, 
        planning_horizon, 
        opt_iters, 
        model, 
        device="cpu",
        nz=16,
        zmin=-np.inf,
        zmax=np.inf,
        nu=2,
        umin=-1.0,
        umax=1.0,
        Q=1.0,
        R=1.0):

        super(CVXLinear, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )

        self.Q = Q * sparse.eye(nz)
        self.R = R * sparse.eye(nu)
        self.nz = nz
        self.zmin = zmin
        self.zmax = zmax
        self.nu = nu
        self.umin = umin
        self.umax = umax
        self.setup_cvx_problem()

    def setup_cvx_problem(self):
        # Define parameters
        self.z_i = cp.Parameter(self.nz)
        self.z_g = cp.Parameter(self.nz)

        self.A = [None] * self.H
        self.B = [None] * self.H

        for h in range(self.H):
            self.A[h] = cp.Parameter((self.nz, self.nz))
            self.B[h] = cp.Parameter((self.nz, self.nu))

        # Define action and observation vectors (variables)
        self.u = cp.Variable((self.H, self.nu))
        self.z = cp.Variable((self.H + 1, self.nz))

        cost = 0
        constraints = [self.z[0] == self.z_i]

        for k in range(self.H):
            cost += \
                cp.quad_form(self.z[k + 1] - self.z_g, self.Q) + \
                cp.quad_form(self.u[k], self.R)
            constraints += [self.z[k + 1] == self.A[k] @ self.z[k] + self.B[k] @ self.u[k]]
            constraints += [self.zmin <= self.z[k], self.z[k] <= self.zmax]
            constraints += [self.umin <= self.u[k], self.u[k] <= self.umax]

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def solve(self, z_0, mu_0, var_0, z_g, u_0=None):
        """
        Args: 
            z_0: the initial sampled state (torch.tensor)
            mu_0: the initial mean of the state (torch.tensor)
            var_0: the initial variance of the state (torch.tensor)
            z_g: the goal state (torch.tensor)
            u_0: the initial guess for the controls (torch.tensor)
        Returns:
            u_0: the final solution (torch.tensor)
        """
        with torch.no_grad():
            if u_0 == None:
                u_0 = torch.zeros(
                    (self.H, 1, self.nu), 
                    device=self.device
                )

            for _ in range(self.opt_iters):

                z_hat, info = self.model.rollout(
                    z_0=z_0, 
                    mu_0=mu_0, 
                    var_0=var_0, 
                    u=u_0
                )

                for h in range(self.H):
                    self.A[h].value = info["A"][h, 0].cpu().numpy()
                    self.B[h].value = info["B"][h, 0].cpu().numpy()
         
                self.z_i.value = z_0.squeeze(0).cpu().numpy()
                self.z_g.value = z_g.cpu().numpy()
 
                self.prob.solve(warm_start=True)

                # Update operational point u_0
                u_0 = self.u.value
                u_0 = np.expand_dims(u_0, axis=1)
                u_0 = torch.tensor(u_0)

            return u_0

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
        device="cpu", 
        nu=2,
        nz=16,
        samples=128, 
        top_samples=32):

        super(CEM, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )
        self.samples = samples
        self.top_samples = top_samples
        self.nu = nu
        self.nz = nz

    def solve(self, z_0, mu_0, var_0, z_g):
        """
        Args: 
            z_0: the initial sampled state (torch.tensor)
            mu_0: the initial mean of the state (torch.tensor)
            var_0: the initial variance of the state (torch.tensor)
            z_g: the goal state (torch.tensor)
        Returns:
            u_0: the final solution (torch.tensor)
        """
        with torch.no_grad():
            if self.samples > 1:
                z_0 = z_0.repeat(self.samples, 1)
                mu_0 = mu_0.repeat(self.samples, 1)
                var_0 = var_0.repeat(self.samples, 1, 1)

            # Initialize factorized belief over action sequences q(u_t:t+H) ~ N(0, I)
            u_mu = torch.zeros(
                self.H, 
                self.nu, 
                device=self.device
            )
            u_mu = u_mu.unsqueeze(1).repeat(1, self.samples, 1)

            u_std = torch.ones(
                self.H, 
                self.nu, 
                device=self.device
            )
            u_std = u_std.unsqueeze(1).repeat(1, self.samples, 1)

            for _ in range(self.opt_iters):
                eps = torch.randn_like(u_std)
                u = u_mu + eps * u_std

                z_hat, info = self.model.rollout(
                    z_0=z_0, 
                    mu_0=mu_0, 
                    var_0=var_0, 
                    u=u
                )          

                cost = (z_g - z_hat).sum(-1)

                # Find top k lowest costs
                _, top_k_idx = cost.topk(
                    self.top_samples, 
                    dim=1, 
                    largest=False
                )

                # Update mean and std
                for ii in range(self.H):
                    u_mu[ii] = u_mu[ii, top_k_idx[ii]].mean(dim=0)
                    u_std[ii] = u_std[ii, top_k_idx[ii]].std(dim=0)
                
        
            
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
        device="cpu", 
        grad_clip=True,
        nu=2,
        nz=16):

        super(Grad, self).__init__(
            planning_horizon=planning_horizon,
            opt_iters=opt_iters,
            model=model,
            device=device
        )
        self.grad_clip = grad_clip
        self.nu = nu
        self.nz = nz
        
    def solve(self, z_0, mu_0, var_0, z_g, u_0=None):
        """
        Args: 
            z_0: the initial sampled state (torch.tensor)
            mu_0: the initial mean of the state (torch.tensor)
            var_0: the initial variance of the state (torch.tensor)
            z_g: the goal state (torch.tensor)
            u_0: the initial guess for the controls (torch.tensor)
        Returns:
            u_0: the final solution (torch.tensor)
        """
        if u_0 == None:
            u_0 = torch.zeros(
                (self.H, 1, self.nu), 
                device=self.device
            )
        u_0 = u_0.clone().detach()
        u_0.requires_grad = True

        #TODO: Perturb guess randomly or perturb and run multiple solves and take best result?

        opt = optim.SGD([u_0], lr=0.1, momentum=0)

        for _ in range(self.opt_iters):
            z_hat, info = self.model.rollout(
                z_0=z_0, 
                mu_0=mu_0, 
                var_0=var_0, 
                u=u_0
            )            
            cost = torch.sum(z_g - z_hat)

            #TODO: Grad clip?
            opt.zero_grad()
            (-cost).backward()
            opt.step()

        return u_0.detach()