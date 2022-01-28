"""
General util methods used.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
from networks import (FullyConvEncoderVAE,
                        FullyConvDecoderVAE,
                        FCNEncoderVAE,
                        FCNDecoderVAE,
                        CNNEncoder1D,
                        CNNDecoder1D,
                        RNNEncoder)
from models import (LinearMixSSM, 
                    NonLinearSSM)
import gym
from collections import deque
from gym import spaces
import os 

def rgb2gray(x):
    """
    Convert pytorch tensor of RGB images (b, h, w, c) to 
    grayscale (b, h, w, c) renormalized between 0 to 1.
    """
    x = np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])[..., np.newaxis]
    x_min, x_max = x.min(), x.max()
    x = (x - x_min) / (x_max - x_min)
    return x

def frame_stack(x, frames=1):
    """
    Given a trajectory of images with shape (n, l, c, ...) convert to 
    (n, l - frames, (frames + 1) * c, ...), where the channel dimension 
    contains the extra frames added.
    
    e.g. visualization of frames=2:
    x_{0} | x_{1} x_{2} x_{3} ... x_{l}   |
    0     | x_{0} x_{1} x_{2} ... x_{l-1} | x_{l}

    NOTE: "Index 0" is the current frame, and index 1+ is the history
    """
    n, l, c = x.shape[:3]
    x_stacked = torch.zeros((n, l, (frames + 1) * c, *x.shape[3:]), 
                                dtype=x.dtype, device=x.device)
    x_stacked[:, :, :c] = x
    for ii in (_ + 1 for _ in range(frames)):
        pad = torch.zeros((n, ii, c, *x.shape[3:]), 
                            dtype=x.dtype, device=x.device)
        x_stacked[:, :, ((ii) * c):((ii+1) * c)] = \
            torch.cat((pad, x), dim=1)[:, :l]
    # slice off the initial part of the traj w/ no history
    x_stacked = x_stacked[:, frames:]
    return x_stacked

def set_seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def common_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # consider also xavier_uniform_, kaiming_uniform_ , orthogonal_
    elif type(m) == nn.Conv1d or type(m) == nn.Conv2d or type(m) == nn.Conv3d:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)
        
def weight_norm(m):
    weight_norm_layers = [
        nn.Linear, 
        nn.Conv1d, 
        nn.Conv2d, 
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]

    if type(m) in weight_norm_layers:
        torch.nn.utils.weight_norm(m)

def load_vh_models(args, path=None, mode='eval', device='cuda:0'):
    """Load the trained visual haptic models based on args."""
    if path is not None:
        print("Loading models in path: ", path)
    
    # Non-linearities for networks
    if args.non_linearity=="relu":
        nl = nn.ReLU()
    elif args.non_linearity=="elu":
        nl = nn.ELU()
    elif args.non_linearity=="softplus":
        nl = nn.Softplus()
    else:
        raise NotImplementedError()

    nets = {}

    # Networks
    nets["img_enc"] = FullyConvEncoderVAE(
        input=args.dim_x[0] * (args.frame_stacks + 1),
        latent_size=args.dim_z_img,
        bn=args.use_batch_norm,
        drop=args.use_dropout,
        nl=nl,
        img_dim=args.dim_x[1],
        stochastic=(args.inference_network == "none")
    ).to(device=device)

    if args.context_modality != "none":
        if args.context_modality == "joint": 
            data_dim = args.dim_arm + args.dim_ft
        elif args.context_modality == "arm": 
            data_dim = args.dim_arm
        elif args.context_modality == "ft":
            data_dim = args.dim_ft

        if args.use_context_frame_stack:
            data_len = 2 * args.context_seq_len
        else:
            data_len = args.context_seq_len

        nets["context_enc"] = CNNEncoder1D(
            input=data_dim,
            datalength=data_len,
            latent_size=args.dim_z_context,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            stochastic=(args.inference_network == "none")
        ).to(device=device)

    if args.inference_network == "none":
        pass
    elif args.inference_network == "ssm":
        nets["img_ssm_enc"] = FCNEncoderVAE(
            dim_in=2*args.dim_z_img,
            dim_out=args.dim_z_img,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            hidden_size=args.fc_hidden_size,
            stochastic=True
        )

        if args.context_modality != "none":
            nets["context_ssm_enc"] = FCNEncoderVAE(
                dim_in=2*args.dim_z_context,
                dim_out=args.dim_z_context,
                bn=args.use_batch_norm,
                drop=args.use_dropout,
                nl=nl,
                hidden_size=args.fc_hidden_size,
                stochastic=True
            )
    elif args.inference_network == "rssm":
        nets["img_rssm_enc"] = RNNEncoder(
            dim_in=args.dim_z_img,
            dim_out=args.dim_z_img,
            train_initial_hidden=args.train_initial_hidden,
            hidden_size=args.rnn_hidden_size,
            stochastic=True
        ).to(device=device)
        
        if args.context_modality != "none":
            nets["context_rssm_enc"] = RNNEncoder(
                dim_in=args.dim_z_context,
                dim_out=args.dim_z_context,
                train_initial_hidden=args.train_initial_hidden,
                stochastic=True
            ).to(device=device)

    if args.use_binary_ce:
        output_nl = None
    else:
        output_nl = nn.Sigmoid()

    nets["img_dec"] = FullyConvDecoderVAE(
        input=args.dim_x[0] * (args.frame_stacks + 1),
        latent_size=args.dim_z_img,
        bn=args.use_batch_norm,
        drop=args.use_dropout,
        nl=nl,
        img_dim=args.dim_x[1],
        output_nl=output_nl
    ).to(device=device)

    if args.context_modality != "none" and args.reconstruct_context:
        nets["context_dec"] = CNNDecoder1D(
            input=data_dim, 
            latent_size=args.dim_z_context, 
            bn=args.use_batch_norm, 
            drop=args.use_dropout, 
            nl=nl, 
            output_nl=None, 
            datalength=data_len
        ).to(device=device)

    # Dynamics network
    if args.dyn_net == "linearmix":
        nets["dyn"] = LinearMixSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            K=args.K,
            train_initial_hidden=args.train_initial_hidden,
            learn_uncertainty=args.learn_uncertainty
        ).to(device=device)
    elif args.dyn_net == "nonlinear":
        nets["dyn"] = NonLinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            train_initial_hidden=args.train_initial_hidden
        ).to(device=device)
    else:
        raise NotImplementedError()

    if args.use_weight_norm:
        for k, v in nets.items():
            v.apply(weight_norm)

    if path is not None:
        for k, model in nets.items():
            try:
                model.load_state_dict(
                    torch.load(path + f"/{k}.pth", map_location=device)
                )
                if mode == 'eval':
                    model.eval()
                elif mode == 'train':
                    model.train()
                else:
                    raise NotImplementedError()
            except Exception as e: 
                print(e)             
    
    return nets