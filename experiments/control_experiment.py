import os, sys
os.sys.path.insert(0, "..")

from mpc import CVXLinear, Grad, CEM
from mpc_wrappers import LinearMixWrapper
from models import LinearMixSSM
import torch
from utils import load_vh_models

def solve_with_all_mpc_variants(z_0, z_g, f, device):
    cvxmpc = CVXLinear(
        planning_horizon=8,
        opt_iters=10,
        model=f,
    )
    cvxmpc.to(device=device)
    u_cvx = cvxmpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    gradmpc = Grad(
        planning_horizon=8,
        opt_iters=100,
        model=f,
    )
    gradmpc.to(device=device)
    u_grad = gradmpc.solve(
        z_0=z_0,
        z_g=z_g
    )

    cem_mpc = CEM(
        planning_horizon=8,
        opt_iters=100,
        model=f,
    )
    cem_mpc.to(device=device)
    u_cem_mu = cem_mpc.solve(
        z_0=z_0,
        z_g=z_g
    )
    
    sol = {
        "cvx": u_cvx,
        "grad": u_grad,
        "cem": u_cem_mu
    }

    return sol

def main():

    # netpath="test"
    # nets = load_vh_models(netpath)

    #TODO: Load newest models
    #TODO: Embed initial and goal image

    # z_0 = {
    #     "z":torch.rand((1, 16)), 
    #     "mu":torch.rand((1, 16)), 
    #     "cov":torch.eye(16).unsqueeze(0)
    # }
    # z_g = torch.rand((16))

    # sol = solve_with_all_mpc_variants(z_0, z_g, f, device="cpu")

    #TODO: pick one solver and send 
if __name__ == "__main__":
    main()