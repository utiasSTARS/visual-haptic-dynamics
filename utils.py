"""
General util methods used.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import random
from models import (FullyConvEncoderVAE,
                    FullyConvDecoderVAE,
                    FCNEncoderVAE,
                    FCNDecoderVAE,
                    LinearMixRNN)

class Normalize:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, x):
        return (x - self.mean) / self.var

    def __repr__(self):
        return self.__class__.__name__ + '(mean={self.mean}, var={self.var})'

def frame_stack(x, frames=1):
    """
    Given a trajectory images with shape (n, l, c, h, w) convert to 
    (n, l - frames, (frames + 1) * c, h, w), where the channel dimension 
    contains the extra frames added.
    
    e.g. visualization of frames=2:
    x_{0} x_{1} x_{2} x_{3} ... x_{l}
    0     x_{0} x_{1} x_{2} ... x_{l-1} x_{l}   
    """
    n, l, c, h, w = x.shape
    x_stacked = torch.zeros((n, l, (frames + 1) * c, h, w), 
                                dtype=x.dtype, device=x.device)
    x_stacked[:, :, :c] = x
    for ii in (_ + 1 for _ in range(frames)):
        pad = torch.zeros((n, ii, c, h, w), 
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
    else:
        raise NotImplementedError()

    if args.enc_dec_net == "fcn":
        enc = FCNEncoderVAE(dim_in=obs_flatten_dim,
                            dim_out=args.dim_z,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            hidden_size=args.fc_hidden_size,
                            stochastic=True).to(device=device)
    elif args.enc_dec_net == "cnn":
        enc = FullyConvEncoderVAE(input=1,
                                    latent_size=args.dim_z,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    img_dim=str(args.dim_x[1]),
                                    nl=nl,
                                    stochastic=True).to(device=device)
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
        dec = FCNDecoderVAE(dim_in=args.dim_z,
                            dim_out=args.dim_x,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            output_nl=output_nl,
                            hidden_size=args.fc_hidden_size).to(device=device)
    elif args.enc_dec_net == "cnn":
        dec = FullyConvDecoderVAE(input=1,
                                  latent_size=args.dim_z,
                                  bn=args.use_batch_norm,
                                  drop=args.use_dropout,
                                  img_dim=str(args.dim_x[1]),
                                  nl=nl,
                                  output_nl=output_nl).to(device=device)
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
    dyn = LinearMixRNN(dim_z=args.dim_z,
                        dim_u=args.dim_u,
                        hidden_size=args.rnn_hidden_size,
                        bidirectional=args.use_bidirectional,
                        net_type=args.rnn_net,
                        K=args.K).to(device=device) 
        
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