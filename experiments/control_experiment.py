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
from datasets import VisualHaptic
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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

def load_goal(idx=None):
    goals = pkl_loader("./goal_imgs/goals.pkl")
    if idx is not None:
        return goals[idx] 
    else:
        return goals[np.random.randint(len(goals))]

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

def generate_loader(dataset):
    dataset_idx = list(range(len(dataset)))
    sampler = SubsetRandomSampler(dataset_idx)
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        sampler=sampler,
    )
    return loader

def control_experiment(args):
    # Load models and env
    model_args = model_arg_loader(args.model_path)

    # Load dataset
    dataset = VisualHaptic(
        args.dataset_path,
        rgb=model_args.dim_x[0] == 3
    )

    # Load initial weights trained with offline dataset
    nets = load_vh_models(
        path=args.model_path, 
        args=model_args, 
        mode='eval', 
        device=args.device
    )

    # Load optimizer
    if args.opt == "adam":
        opt_type = torch.optim.Adam
    elif args.opt == "sgd":
        opt_type = torch.optim.SGD
    else:
        raise NotImplementedError()

    params = [list(v.parameters()) for k, v in nets.items() if "enc" in k]
    params = [v for sl in params for v in sl] # remove nested list

    opt = opt_type(
        params, 
        lr=args.lr
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
        if new_episodes % args.n_train_episodes == 0:
            for k, v in nets.items():
                v.train()
            loader = generate_loader(dataset)
            
            for _ in args.n_epochs:
                summary = opt_iter(
                    epoch=epoch, 
                    loader=loader, 
                    nets=nets, 
                    device=args.device,
                    opt=opt
                )
                epoch += 1

            for k, v in nets.items():
                v.eval()
            new_episodes = 1

        env.reset()
        
        # Load goal
        img_g, context_data_g, gt_plate_pos_g = load_goal()
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

        episode_data = {
            "img": np.zeros((1, 16, 64, 64, 3), dtype=np.uint8), 
            "ft": np.zeros((1, 16, 32, 6)), 
            "arm": np.zeros((1, 16, 32, 6)),
            "action": np.zeros((1, 16, 2)), 
            "gt_plate_pos": np.zeros((1, 16, 3)),
        }

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
                opt=args.mpc_opt, 
                horizon=args.H
            )
            u = sol[0,0].detach().cpu().numpy()
            print("controls :", u)

            # Send control input one time step (n=1)
            obs_tpn, reward, done, info = env.step(u)

            # Store transition in episode data
            episode_data["img"][0, ii] = obs_tpn["img"]
            episode_data["ft"][0, ii] = obs_tpn["ft"]
            episode_data["arm"][0, ii] = obs_tpn["arm"]
            episode_data["action"][0, ii] = u
            episode_data["gt_plate_pos"][0, jj] = info["achieved_goal"] 

            # Initialize hidden state with previous step
            f.set_nstep_hidden_state()

            # Updated state
            img_tpn, context_data_tpn = format_obs(obs_tpn, device=args.device)
            img_hist.appendleft(img_tpn)
            img_i = torch.cat(tuple(img_hist), dim=1)
            context_data_i = context_data_tpn

            # Update initial guess
            u_0[:-1] = sol[1:]
        
        dataset.append(episode_data)
        new_episodes += 1

def main():
    args = parse_control_experiment_args()
    control_experiment(args)

if __name__ == "__main__":
    main()