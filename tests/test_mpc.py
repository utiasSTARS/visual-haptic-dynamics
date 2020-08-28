import os, sys
os.sys.path.insert(0, "..")

from mpc import CVXLinear, Grad, CEM
from mpc_wrappers import LinearStateSpaceWrapper, LinearMixWrapper
from models import LinearMixSSM
import numpy as np
import torch
import scipy.sparse as sparse


def random_MPC_test():
    dyn = LinearMixSSM(
        dim_z=16,
        dim_u=2,
        hidden_size=255,
        bidirectional=False,
        net_type="gru",
        K=15
    ).to(device="cpu").eval()
    for param in dyn.parameters():
        param.requires_grad = False
    dyn_wrapped = LinearMixWrapper(
        dyn_model=dyn
    )
    dyn_wrapped.to(device="cpu")

    z = torch.rand((1, 16))
    mu = torch.rand((1, 16))
    var = torch.eye(16).unsqueeze(0)
    z_0 = {"z":z, "mu":mu, "cov":var}
    z_g = torch.rand((16))

    cvx_mpc = CVXLinear(
        planning_horizon=8,
        opt_iters=1,
        model=dyn_wrapped,
    )
    cvx_mpc.to(device="cpu")
    u_cvx = cvx_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    grad_mpc = Grad(
        planning_horizon=8,
        opt_iters=100,
        model=dyn_wrapped,
    )
    grad_mpc.to(device="cpu")
    u_grad = grad_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    cem_mpc = CEM(
        planning_horizon=8,
        opt_iters=10,
        model=dyn_wrapped,
    )
    cem_mpc.to(device="cpu")
    u_cem_mu = cem_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    # print("grad", u_grad)
    # print("cvx", u_cvx)
    # print("cem", u_cem_mu)

def cvxpy_simple_MPC_example_test():
    # Generate data for control problem.
    np.random.seed(1)
    n = 8
    m = 2
    T = 50
    alpha = 0.2
    beta = 5
    A = np.eye(n) + alpha*np.random.randn(n,n)
    B = np.random.randn(n,m)
    z_0 = beta * np.random.randn(n)

    A = torch.from_numpy(A).repeat(T, 1, 1, 1).float()
    B = torch.from_numpy(B).repeat(T, 1, 1, 1).float()
    z_0 = torch.from_numpy(z_0).repeat(1, 1).float()
    z_g = torch.zeros((n)).float()

    dyn_wrapped = LinearStateSpaceWrapper(
        A=A,
        B=B
    )
    dyn_wrapped.to(device="cpu")

    cvx_mpc = CVXLinear(
        planning_horizon=T,
        opt_iters=1,
        nz=n,
        nu=m,
        model=dyn_wrapped,
    )
    cvx_mpc.to(device="cpu")
    u_cvx = cvx_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    grad_mpc = Grad(
        planning_horizon=T,
        opt_iters=100,
        nz=n,
        nu=m,
        model=dyn_wrapped,
    )
    grad_mpc.to(device="cpu")
    u_grad = grad_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    cem_mpc = CEM(
        planning_horizon=T,
        opt_iters=20,
        nz=n,
        nu=m,
        model=dyn_wrapped,
    )
    cem_mpc.to(device="cpu")
    u_cem_mu = cem_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    # print("grad", u_grad)
    # print("cvx", u_cvx)
    # print("cem", u_cem_mu)

if __name__ == "__main__":
    random_MPC_test()
    print("=" * 89)
    cvxpy_simple_MPC_example_test()