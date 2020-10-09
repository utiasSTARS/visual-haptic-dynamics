import os, sys
os.sys.path.insert(0, "..")
import inspect

from mpc import CVXLinear, Grad, CEM
from mpc_wrappers import LinearMixWrapper
from models import LinearMixSSM
from utils import load_vh_models

import torch
import numpy as np

from args.parser import parse_control_experiment_args
from argparse import Namespace
import json
import pickle as pkl

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir + "/pixel-environments/gym_thing/gym_thing/")
from visual_pusher_env import ThingVisualPusher
import pybullet as p

def model_arg_loader(path):
    """Load hyperparameters from trained model."""
    model = {}
    if os.path.isdir(path):
        with open(os.path.join(path, 'hyperparameters.txt'), 'r') as fp:
            model[path] = Namespace(**json.load(fp))
    return model

def pkl_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def load_env(args):
    # Load environment config for dataset which model was trained on
    dataset_name = args.dataset.split("/")[-1]
    idx = dataset_name.index(".pkl")
    dataset_config = dataset_name[:idx] + "_config" + dataset_name[idx:]
    env_config = pkl_loader("./data/datasets/" + dataset_config)
    
    # Rendering
    env_config["is_render"] = True

    #Load env with right config
    env = ThingVisualPusher.from_config(env_config)
    return env

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

def control_experiment(args):
    model_dict = model_arg_loader(args.model_path)
    for path, model_args in model_dict.items():
        nets = load_vh_models(path=path, args=model_args, mode='eval', device=args.device)
        env = load_env(args=model_args)

        for ii in range(10000):
            env.step(np.array([0,0]))
            print(ii)
    
        #TODO: Embed initial and goal image


        # z_0 = {
        #     "z":torch.rand((1, 16)), 
        #     "mu":torch.rand((1, 16)), 
        #     "cov":torch.eye(16).unsqueeze(0)
        # }
        # z_g = torch.rand((16))

        # sol = solve_with_all_mpc_variants(z_0, z_g, f, device="cpu")

        #TODO: pick one solver and send 

def main():
    args = parse_control_experiment_args()
    control_experiment(args)

if __name__ == "__main__":
    main()