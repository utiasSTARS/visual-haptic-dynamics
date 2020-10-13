import os, sys
os.sys.path.insert(0, "..")
import inspect
from collections import deque  

from mpc import CVXLinear, Grad, CEM
from mpc_wrappers import LinearMixWrapper
from models import LinearMixSSM
from utils import load_vh_models, rgb2gray

import torch
import numpy as np
import matplotlib.pyplot as plt

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
    if os.path.isdir(path):
        with open(os.path.join(path, 'hyperparameters.txt'), 'r') as fp:
            return Namespace(**json.load(fp))

def pkl_loader(path):
    """A generic data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def load_env(args):
    """Load environment with config from dataset which model was trained on."""
    dataset_name = args.dataset.split("/")[-1]
    idx = dataset_name.index(".pkl")
    dataset_config = dataset_name[:idx] + "_config" + dataset_name[idx:]
    env_config = pkl_loader("./data/datasets/" + dataset_config)
    
    # Rendering set as true for experiments
    env_config["is_render"] = True

    #Load env with right config
    env = ThingVisualPusher.from_config(env_config)

    return env

def load_goal(idx):
    goals = pkl_loader("./goal_imgs/goals.pkl")
    return goals[idx] 

def format_obs(obs, device="cpu"):
    """Format obs from env for network."""
    # Convert to tensor
    img = torch.tensor(rgb2gray(obs["img"]), device=device)
    context_data = torch.cat((
        torch.tensor(obs["ft"], device=device), 
        torch.tensor(obs["arm"], device=device)
    ), dim=-1)

    # Reformat in right shape
    img = img.permute(2, 0, 1).unsqueeze(0)
    context_data = context_data.transpose(1, 0).unsqueeze(0)

    return img, context_data

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
    
    # Load models and env
    model_args = model_arg_loader(args.model_path)
    nets = load_vh_models(path=args.model_path, args=model_args, mode='eval', device=args.device)
    env = load_env(args=model_args)
    
    # Keep track of history of state using deque
    img_hist = deque([], (1 + model_args.frame_stacks))

    # Format initial image for network in method
    obs_t = env.reset()
    img_t, context_data_t = format_obs(obs_t, device=args.device)
    img_hist.appendleft(img_t)

    # Step randomly to produce a history
    obs_tp1, _, _, _ = env.step(env.action_space.sample())
    img_tp1, context_data_tp1 = format_obs(obs_tp1, device=args.device)
    img_hist.appendleft(img_tp1)
    
    # Initial state
    img_i = torch.cat(tuple(img_hist), dim=1)
    context_data_i = context_data_tp1

    # Load goal
    img_g, context_data_g, gt_plate_pos_g = load_goal(444)

    print(img_i.shape, img_g.shape)
    print(context_data_i.shape, context_data_g.shape)
    
    # for ii in range(10000):

        #TODO: Embed initial image 
        #img torch.Size([1600, 1, 2, 64, 64])
        #ctx img torch.Size([1600, 2, 64, 64])


        #TODO: Solve MPC
        # z_0 = {
        #     "z":torch.rand((1, 16)), 
        #     "mu":torch.rand((1, 16)), 
        #     "cov":torch.eye(16).unsqueeze(0)
        # }
        # z_g = torch.rand((16))

        # sol = solve_with_all_mpc_variants(z_0, z_g, f, device="cpu")

        #TODO: pick one solver and send 

        # env.step(np.array([0,0]))
        # print(ii)


def main():
    args = parse_control_experiment_args()
    control_experiment(args)

if __name__ == "__main__":
    main()