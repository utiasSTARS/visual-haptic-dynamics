import os, sys
os.sys.path.insert(0, "..")

from mpc import CVXLinear, Grad, CEM, LinearMixWrapper
from models import LinearMixSSM
import numpy as np
import torch

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

u_cem = cem_mpc = CEM(
    planning_horizon=12,
    opt_iters=10,
    model=dyn_wrapped,
)
cem_mpc.to(device="cpu")

u_cvx = cvx_mpc = CVXLinear(
    planning_horizon=12,
    opt_iters=10,
    model=dyn_wrapped,
)
cvx_mpc.to(device="cpu")

u_grad = grad_mpc = Grad(
    planning_horizon=12,
    opt_iters=10,
    model=dyn_wrapped,
)
grad_mpc.to(device="cpu")

z = torch.zeros((1, 16))
mu = torch.zeros((1, 16))
var = torch.zeros((1, 16, 16))
z_0 = {"z":z, "mu":mu, "cov":var}
z_g = torch.zeros((16))

u_cvx = cvx_mpc.solve(
    z_0=z_0,
    z_g=z_g
)

u_grad = grad_mpc.solve(
    z_0=z_0,
    z_g=z_g
)

(u_cem_mu, u_cem_std) = cem_mpc.solve(
    z_0=z_0,
    z_g=z_g
)