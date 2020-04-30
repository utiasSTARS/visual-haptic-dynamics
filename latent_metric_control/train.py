from utils import (set_seed_torch, common_init_weights)
set_seed_torch(3)
def _init_fn(worker_id):
    np.random.seed(int(3))
import numpy as np
from args.parser import parse_training_args
from collections import OrderedDict
import json
import os, sys, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from models import (FullyConvEncoderVAE,
                    FullyConvDecoderVAE,
                    FCNEncoderVAE,
                    FCNDecoderVAE,
                    LinearMixRNN)
from datasets import ImgCached

def loop(args):
    assert 0 <= args.opt_vae_epochs <= args.opt_vae_base_epochs <= args.n_epoch
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # Keeping track of results and hyperparameters
    if not args.debug:
        time_tag = datetime.strftime(datetime.now(), '%m-%d-%y_%H:%M:%S')
        save_dir = args.storage_base_path + time_tag + '_' + args.comment
        os.makedirs(save_dir, exist_ok=True)

        if args.n_epoch > args.n_checkpoint_epoch:
            checkpoint_dir = os.path.join(save_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

        args.__dict__ = OrderedDict(sorted(args.__dict__.items(), key=lambda t: t[0]))
        with open(save_dir + '/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
        writer = SummaryWriter(logdir=save_dir)

    # Non-linearities for networks
    if args.non_linearity=="relu":
        nl = nn.ReLU()
    elif args.non_linearity=="elu":
        nl = nn.ELU()
    else:
        raise NotImplementedError()
    output_nl = None if args.use_binary_ce else nn.Sigmoid()

    # Encoder and decoder
    if args.enc_dec_net == 'fcn':
        enc = FCNEncoderVAE(dim_in=int(np.product(args.dim_x)),
                            dim_out=args.dim_z,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            hidden_size=args.fc_hidden_size,
                            stochastic=True).to(device=device)
        dec = FCNDecoderVAE(dim_in=args.dim_z,
                            dim_out=args.dim_x,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            output_nl=output_nl,
                            hidden_size=args.fc_hidden_size).to(device=device)
    elif args.enc_dec_net == 'cnn':
        enc = FullyConvEncoderVAE(input=1,
                                    latent_size=args.dim_z,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    img_dim=str(args.dim_x[1]),
                                    stochastic=True).to(device=device)
        dec = FullyConvDecoderVAE(input=1,
                                    latent_size=args.dim_z,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    img_dim=str(args.dim_x[1]),
                                    output_nl=output_nl).to(device=device)

    # Dynamics network
    dyn = LinearMixRNN(input_size=args.dim_alpha,
                        dim_z=args.dim_z,
                        dim_u=args.dim_u
                        hidden_size=args.rnn_hidden_size,
                        bidirectional=args.use_bidirectional,
                        net_type=args.rnn_net,
                        K=args.K)
    base_matrices_param = [dyn.A, dyn.B, dyn.C]

    if args.opt == "adam":
        opt_vae = torch.optim.Adam(list(enc.parameters()) + 
                                   list(dec.parameters()), 
                                   lr=args.lr)
        opt_vae_base = torch.optim.Adam(list(enc.parameters()) + 
                                        list(dec.parameters()) + 
                                        base_matrices_param, 
                                        lr=args.lr)
        opt_all = torch.optim.Adam(list(enc.parameters()) + 
                                   list(dec.parameters()) +
                                   list(dyn.parameters()) + 
                                   base_matrices_params, 
                                   lr=args.lr)
    elif args.opt == "sgd":
        opt_vae = torch.optim.SGD(list(enc.parameters()) + 
                                  list(dec.parameters()), 
                                  lr=args.lr, 
                                  momentum=0.9, 
                                  nesterov=True)
        opt_vae_base = torch.optim.SGD(list(enc.parameters()) + 
                                     list(dec.parameters()) + 
                                     base_matrices_param,
                                     lr=args.lr, 
                                     momentum=0.9, 
                                     nesterov=True)
        opt_all = torch.optim.SGD(list(enc.parameters()) + 
                                  list(dec.parameters()) +
                                  list(dyn.parameters()) + 
                                  base_matrices_params, 
                                  lr=args.lr, 
                                  momentum=0.9, 
                                  nesterov=True)
    else:
        raise NotImplementedError()

    if args.weight_init == 'custom':
        enc.apply(common_init_weights)
        dec.apply(common_init_weights)
        lgssm.apply(common_init_weights)
        if args.measurement_uncertainty == 'feature': 
            R_net.apply(common_init_weights)
    
    # Loss functions
    if args.use_binary_ce:
        loss_REC = nn.BCEWithLogitsLoss(reduction='none').to(device=device)
    else:
        loss_REC = nn.MSELoss(reduction='none').to(device=device)

    if args.task == "pendulum64":
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            Normalize(mean=0.27, var=1.0 - 0.27) # 64x64
            ])
    else:
        raise NotImplementedError()

    # Dataset
    ds = ImgCached(args.dataset,
                   transform=transform,
                   img_shape=args_dim_x)
    ds_size = len(ds)
    idx = list(range(ds_size))
    split = int(np.floor(args.val_split * ds_size))
    train_idx, val_idx = idx[split:], idx[:split]
    train_spler = SubsetRandomSampler(train_idx)
    valid_spler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(ds,
                              batch_size=args.n_batch,
                              num_workers=args.n_worker,
                              sampler=train_spler,
                              worker_init_fn=_init_fn)
    val_loader = DataLoader(ds,
                            batch_size=args.n_batch,
                            num_workers=args.n_worker,
                            sampler=valid_spler,
                            worker_init_fn=_init_fn)

    def opt_iter(epoch, opt=None):
        """Single training epoch."""
        if opt:
            enc.train()
            dec.train()
            dyn.train()
            loader = train_loader
        else:
            enc.eval()
            dec.eval()
            dyn.eval()
            loader = val_loader

        avg_l = []
        for idx, data in enumerate(loader):
            if idx == args.n_example:
                break
            
            # Load and shape trajectory data
            # XXX: all trajectories have same length
            x_full = data['images'].float().to(device=device)
            # Sample random range of traj_len
            s_idx = np.random.randint(x_full.shape[1] - args.traj_len + 1)
            e_idx = s_idx + args.traj_len
            x = x_full[:, s_idx:(e_idx - 1)]
            x_dim = x.shape
            n = x_dim[0]
            l = x_dim[1]
            # Reshape to (n * l, 1, height, width)
            x = x.reshape(n * l, *x_dim[2:])
            u = data['actions'].float().to(device=device)[:, (s_idx + 1):e_idx]

            # Encode & Decode sample
            z, _, _ = enc(x)
            x_hat = dec(z)

            z = z.reshape(n, l, args.dim_z)

            #TODO: Forward loss w/ analytical KL
            total_loss = (args.lam_rec * loss_rec + 
                            annealing_factor_beta * args.lam_kl * loss_KL) / N

            avg_l.append((loss_rec.item() + loss_KL.item()) / N)

            # Jointly optimize everything
            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                if args.measurement_uncertainty == 'feature':  
                    torch.nn.utils.clip_grad_norm_(R_net.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(lgssm.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 0.5)
                opt.step()

        avg_loss = sum(avg_l) / len(avg_l)
        return avg_loss

    # Training loop
    opt = opt_vae
    try:
        for epoch in range(0, args.n_epoch):
            tic = time.time()
            if epoch >= args.opt_vae_base_epochs:
                opt = opt_all
                if args.scheduler != 'none':
                    lr_scheduler = lr_scheduler_all
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_base
                if args.scheduler != 'none':
                    lr_scheduler = lr_scheduler_vae_kf
            
            # Train for one epoch        
            avg_train_loss = opt_iter(epoch=epoch, opt=opt)

            # Calculate validtion loss
            if args.val_split > 0:
                with torch.no_grad():
                    avg_val_loss = opt_iter(epoch=epoch)
            else:
                avg_val_loss = 0
            epoch_time = time.time() - tic

            print("Epoch {}/{}: Avg train loss: {}, \
                   Avg val loss: {}, Time per epoch: {}"
                   .format(epoch + 1, ini_epoch + args.n_epoch, 
                   avg_train_loss, avg_val_loss, epoch_time))
            
            # Tensorboard
            if not args.debug:
                writer.add_scalars("Loss", 
                                   {'train': avg_train_loss, 
                                   'val': avg_val_loss}, 
                                   epoch)
            
            # Save model at intermittent checkpoints 
            if (epoch + 1) % args.n_checkpoint_epoch == 0:
                checkpoint_i_path = os.path.join(checkpoint_dir, str((epoch + 1) // args.n_checkpoint_epoch))
                os.makedirs(checkpoint_i_path, exist_ok=True)

                # Save models
                torch.save(dyn.state_dict(), checkpoint_i_path + '/dyn.pth')
                torch.save(enc.state_dict(), checkpoint_i_path + '/enc.pth')
                torch.save(dec.state_dict(), checkpoint_i_path + '/dec.pth')

    finally:
        if not args.debug:
            if not np.isnan(avg_train_loss):
                # Save models
                torch.save(dyn.state_dict(), data_dir + '/dyn.pth')
                torch.save(enc.state_dict(), data_dir + '/enc.pth')
                torch.save(dec.state_dict(), data_dir + '/dec.pth')
            writer.close()

def main():
    args = parse_training_args()
    loop(args)

if __name__ == "__main__":
    main()
