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
    lgssm = LGSSM(dim_z=args.dim_z,
                    dim_a=args.dim_a,
                    dim_u=args.dim_u,
                    alpha_net=alpha_net,
                    K=args.k,
                    transition_noise=args.transition_noise,
                    emission_noise=args.emission_noise,
                    device=device).to(device=device)    
        
    try:
        lgssm.load_state_dict(torch.load(path + "/lgssm.pth", map_location=device))
        if mode == 'eval':
            lgssm.eval()
        elif mode == 'train':
            lgssm.train()
        else:
            raise NotImplementedError()
    except Exception as e: 
        print(e)             
    
    return enc, dec, lgssm