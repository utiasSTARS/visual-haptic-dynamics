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
                        CNNEncoder1D)
from models import (LinearMixSSM, 
                    LinearSSM, 
                    NonLinearSSM)
import gym
from collections import deque
from gym import spaces

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(self.frames)

class Normalize:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, x):
        return (x - self.mean) / self.var

    def __repr__(self):
        return self.__class__.__name__ + '(mean={self.mean}, var={self.var})'

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
        
def to_img(x, shape):
    assert len(shape) == 2
    sig = nn.Sigmoid()
    x = sig(x)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, *shape)
    return x

def common_init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        # consider also xavier_uniform_, kaiming_uniform_ , orthogonal_
    elif type(m) == nn.Conv2d or type(m) == nn.Conv3d:
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif type(m) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(m.weight_hh_l0)
        nn.init.xavier_uniform_(m.weight_ih_l0)
        nn.init.zeros_(m.bias_hh_l0)
        nn.init.zeros_(m.bias_ih_l0)

def load_models(path, args, mode='eval', device='cuda:0'):
    """Load the trained models based on args."""
    print("Loading models in path: ", path)
    obs_flatten_dim = int(np.product(args.dim_x))  

    if args.non_linearity=="relu":
        nl = nn.ReLU()
    elif args.non_linearity=="elu":
        nl = nn.ELU()
    elif args.non_linearity=="softplus":
        nl = nn.Softplus()
    else:
        raise NotImplementedError()

    if args.enc_dec_net == "fcn":
        true_dim_x = args.dim_x[0] + args.frame_stacks, args.dim_x[1], args.dim_x[2]
        enc = FCNEncoderVAE(
            dim_in=int(np.product(true_dim_x)),
            dim_out=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            hidden_size=args.fc_hidden_size,
            stochastic=True
        ).to(device=device)
    elif args.enc_dec_net == "cnn":
        enc = FullyConvEncoderVAE(
            input=args.dim_x[0] + args.frame_stacks,
            latent_size=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            img_dim=args.dim_x[1],
            nl=nl,
            stochastic=True
        ).to(device=device)
    else:
        raise NotImplementedError()
        
    try:
        enc.load_state_dict(torch.load(path + "/enc.pth", map_location=device))
        if mode == 'eval':
            enc.eval()
        elif mode == 'train':
            enc.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)            
        
    output_nl = None if args.use_binary_ce else nn.Sigmoid()
    
    if args.enc_dec_net == "fcn":
        dec = FCNDecoderVAE(
            dim_in=args.dim_z,
            dim_out=true_dim_x,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            output_nl=output_nl,
            hidden_size=args.fc_hidden_size
        ).to(device=device)
    elif args.enc_dec_net == "cnn":
        dec = FullyConvDecoderVAE(
            input=args.dim_x[0] + args.frame_stacks,
            latent_size=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            img_dim=args.dim_x[1],
            nl=nl,
            output_nl=output_nl
        ).to(device=device)
    else:
        raise NotImplementedError()
        
    try:
        dec.load_state_dict(torch.load(path + "/dec.pth", map_location=device))
        if mode == 'eval':
            dec.eval()
        elif mode == 'train':
            dec.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)            

    # Dynamics network
    if args.dyn_net == "linearmix":
        dyn = LinearMixSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net,
            K=args.K
        ).to(device=device)
    elif args.dyn_net == "linearrank1":
        dyn = LinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
    elif args.dyn_net == "nonlinear":
        dyn = NonLinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
    else:
        raise NotImplementedError()

    try:
        dyn.load_state_dict(torch.load(path + "/dyn.pth", map_location=device))
        if mode == 'eval':
            dyn.eval()
        elif mode == 'train':
            dyn.train()
        else:
            raise NotImplementedError()
        
    except Exception as e: 
        print(e)             
    
    return enc, dec, dyn

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
    z_dim_in = 0

    # Networks
    img_enc = FullyConvEncoderVAE(
        input=args.dim_x[0] * (args.frame_stacks + 1),
        latent_size=args.dim_z_img,
        bn=args.use_batch_norm,
        drop=args.use_dropout,
        nl=nl,
        img_dim=args.dim_x[1],
        stochastic=False
    ).to(device=device)
    nets["img_enc"] = img_enc
    z_dim_in += args.dim_z_img

    if args.context_modality != "none":
        if args.context_modality == "joint": 
            data_dim = 12
        elif args.context_modality == "arm" or "ft": 
            data_dim=6

        context_enc = CNNEncoder1D(
            input=data_dim,
            latent_size=args.dim_z_context,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            stochastic=False
        ).to(device=device)
        nets["context_enc"] = context_enc
        z_dim_in += args.dim_z_context

    if args.use_context_img:
        context_img_enc = FullyConvEncoderVAE(
            input=args.dim_x[0] * (args.frame_stacks + 1),
            latent_size=args.dim_z_context,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            img_dim=args.dim_x[1],
            stochastic=False
        ).to(device=device)
        nets["context_img_enc"] = context_img_enc
        z_dim_in += args.dim_z_context

        if args.reconstruct_context_img:
            context_img_dec = FullyConvDecoderVAE(
                input=args.dim_x[0] * (args.frame_stacks + 1),
                latent_size=args.dim_z_context,
                bn=args.use_batch_norm,
                drop=args.use_dropout,
                nl=nl,
                img_dim=args.dim_x[1],
                output_nl=None if args.use_binary_ce else nn.Sigmoid()
            ).to(device=device)
            nets["context_img_dec"] = context_img_dec

    dim_z_rec = args.dim_z

    if args.context_modality != "none":
        dim_z_rec += args.dim_z_context
    if args.use_context_img:
        dim_z_rec += args.dim_z_context

    img_dec = FullyConvDecoderVAE(
        input=args.dim_x[0] * (args.frame_stacks + 1),
        latent_size=dim_z_rec,
        bn=args.use_batch_norm,
        drop=args.use_dropout,
        nl=nl,
        img_dim=args.dim_x[1],
        output_nl=None if args.use_binary_ce else nn.Sigmoid()
    ).to(device=device)
    nets["img_dec"] = img_dec

    mix = FCNEncoderVAE(
        dim_in=z_dim_in,
        dim_out=args.dim_z,
        bn=args.use_batch_norm,
        drop=args.use_dropout,
        nl=nl,
        hidden_size=args.fc_hidden_size,
        stochastic=True
    ).to(device=device)
    nets["mix"] = mix

    # Dynamics network
    if args.dyn_net == "linearmix":
        dyn = LinearMixSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net,
            K=args.K
        ).to(device=device)
    elif args.dyn_net == "linearrank1":
        dyn = LinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
    elif args.dyn_net == "nonlinear":
        dyn = NonLinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
    else:
        raise NotImplementedError()
    nets["dyn"] = dyn
    
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