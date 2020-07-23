import os, sys
os.sys.path.insert(0, "..")

from mpc import CVXLinear, LinearMixWrapper
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
).to(device="cpu")

dyn_wrapped = LinearMixWrapper(
    dyn_model=dyn
)

mpc = CVXLinear(
    planning_horizon=12,
    opt_iters=10,
    model=dyn_wrapped,
)

mpc.to(device="cpu")

z_0 = torch.zeros((1, 16))
mu_0 = torch.zeros((1, 16))
var_0 = torch.zeros((1, 16, 16))
z_g = torch.zeros((16))

mpc.solve(
    z_0=z_0, 
    mu_0=mu_0, 
    var_0=var_0, 
    z_g=z_g
)