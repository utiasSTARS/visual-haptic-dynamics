import os, sys
os.sys.path.insert(0, "..")
import inspect
from collections import deque  

from mpc import CVXLinear, Grad, CEM
from mpc_wrappers import LinearMixWrapper
from models import LinearMixSSM
from utils import load_vh_models, rgb2gray
from train import setup_opt_iter

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
    img = torch.tensor(rgb2gray(obs["img"]), device=device).float()
    context_data = torch.cat((
        torch.tensor(obs["ft"], device=device) / 100.0, 
        torch.tensor(obs["arm"], device=device)
    ), dim=-1).float()

    # Reformat in right shape
    img = img.permute(2, 0, 1).unsqueeze(0)
    context_data = context_data.transpose(1, 0).unsqueeze(0)

    return img, context_data

def encode(nets, model_args, img, ctx, ctx_img):
    z_all_enc = []
    z_img = nets["img_enc"](img)
    z_all_enc.append(z_img)  

    if model_args.context_modality != "none":
        z_context = nets["context_enc"](ctx)
        z_all_enc.append(z_context)   

    if model_args.context in ["initial_latent_state", "goal_latent_state"]:
        z_img_context = nets["context_img_enc"](ctx_img)
        z_all_enc.append(z_img_context)        

    z_cat_enc = torch.cat(z_all_enc, dim=-1)
    z, mu_z, logvar_z = nets["mix"](z_cat_enc)
    var_z = torch.diag_embed(torch.exp(logvar_z))
    
    return z, mu_z, var_z

def solve_mpc(z_0, z_g, u_0, f, device, opt, horizon):
    if opt == "cvxopt":
        cvxmpc = CVXLinear(
            planning_horizon=horizon,
            opt_iters=16,
            model=f,
        )
        cvxmpc.to(device=device)
        u = cvxmpc.solve(
            z_0=z_0,
            z_g=z_g,
            u_0=u_0
        )
    elif opt == "grad":
        gradmpc = Grad(
            planning_horizon=horizon,
            opt_iters=256,
            model=f,
        )
        gradmpc.to(device=device)
        u = gradmpc.solve(
            z_0=z_0,
            z_g=z_g,
            u_0=u_0
        )
    elif opt == "cem":
        cem_mpc = CEM(
            planning_horizon=horizon,
            opt_iters=128,
            model=f,
        )
        cem_mpc.to(device=device)
        u = cem_mpc.solve(
            z_0=z_0,
            z_g=z_g
        )
    return u

def control_experiment(args):
    # Load models and env
    model_args = model_arg_loader(args.model_path)

    # Load initial weights trained with offline dataset
    nets = load_vh_models(
        path=args.model_path, 
        args=model_args, 
        mode='eval', 
        device=args.device
    )

    # Wrap dynamics for MPC code format
    f = LinearMixWrapper(
        nets["dyn"],
        nstep_store_hidden=1,
        device=args.device
    )

    env = load_env(args=model_args)

    new_episodes = 1
    epoch = model_args.n_epoch
    opt_iter = setup_opt_iter(model_args)

    for _ in range(args.n_episodes):
        
        if total_episodes % args.n_train_episodes == 0:
            for k, v in nets.items():
                v.train()

            for _ in args.n_epochs:
                summary_train = opt_iter(
                    epoch=epoch, 
                    loader=train_loader, 
                    nets=nets, 
                    device=device,
                    opt=opt
                )
                epoch += 1

            for k, v in nets.items():
                v.eval()
            new_episodes = 1

        env.reset()
        
        # Load goal
        img_g, context_data_g, gt_plate_pos_g = load_goal(404)
        plt.imshow(img_g[0,0], cmap="gray")
        plt.title("Image Goal")
        plt.show(block=False)

        # Keep track of history of state using deque
        img_hist = deque([], (1 + model_args.frame_stacks))

        # Step to produce a history
        for jj in range(10):
            obs_tp1, _, _, _ = env.step(np.array([0.35, 0]))
            img_tp1, context_data_tp1 = format_obs(obs_tp1, device=args.device)
            img_hist.appendleft(img_tp1)
        
        # Initial state
        img_i = torch.cat(tuple(img_hist), dim=1)
        context_data_i = context_data_tp1

        # Context image based on model type
        if model_args.context in ["initial_latent_state", "initial_image"]:
            ctx_img = img_i
        elif model_args.context in ["goal_latent_state", "goal_image"]:
            ctx_img = img_g
        else:
            ctx_img = None

        # Embed goal image
        with torch.no_grad():
            z_g, mu_z_g, var_z_g = encode(
                nets, 
                model_args, 
                img_g, 
                context_data_g, 
                ctx_img
            )

        # Initial guess
        u_0 = torch.zeros(
            (args.H, 1, 2), 
            device=args.device
        ) + 0.01
        
        for ii in range(16):
            # Embed initial image 
            with torch.no_grad():
                z_i, mu_z_i, var_z_i = encode(
                    nets, 
                    model_args, 
                    img_i, 
                    context_data_i, 
                    ctx_img
                )
            print("Initial distance to latent goal: ", torch.sum((mu_z_g - mu_z_i)**2))

            # Solve MPC problem
            q_i = {
                "z":z_i, 
                "mu":mu_z_i, 
                "cov":var_z_i
            }
            sol = solve_mpc(
                q_i, 
                z_g[0], 
                u_0, 
                f, 
                device=args.device, 
                opt=args.opt, 
                horizon=args.H
            )
            u = sol[0,0].detach().cpu().numpy()
            print("controls :", u)

            # Send control input one time step (n=1)
            obs_tpn, reward, done, info = env.step(u)

            print(
                obs_tpn["img"].shape, 
                obs_tpn["ft"].shape, 
                obs_tpn["arm"].shape, 
                u.shape,
                info["achieved_goal"].shape
            )

            # Initialize hidden state with previous step
            f.set_nstep_hidden_state()

            # Updated state
            img_tpn, context_data_tpn = format_obs(obs_tpn, device=args.device)
            img_hist.appendleft(img_tpn)
            img_i = torch.cat(tuple(img_hist), dim=1)
            context_data_i = context_data_tpn

            # Update initial guess
            u_0[:-1] = sol[1:]
        new_episodes += 1

def main():
    args = parse_control_experiment_args()
    control_experiment(args)

if __name__ == "__main__":
    main()