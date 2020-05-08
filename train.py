from utils import (set_seed_torch, 
                    common_init_weights, 
                    Normalize,
                    frame_stack)
set_seed_torch(3)
def _init_fn(worker_id):
    np.random.seed(int(3))
import numpy as np
from args.parser import parse_training_args
from collections import OrderedDict
from datetime import datetime
from tensorboardX import SummaryWriter
import json
import os, sys, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from models import (FullyConvEncoderVAE,
                    FullyConvDecoderVAE,
                    FCNEncoderVAE,
                    FCNDecoderVAE,
                    LinearMixRNN)
from datasets import ImgCached
from losses import kl

def train(args):
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
        true_dim_x = args.dim_x[0] + args.frame_stacks, args.dim_x[1], args.dim_x[2]
        enc = FCNEncoderVAE(dim_in=int(np.product(true_dim_x)),
                            dim_out=args.dim_z,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            hidden_size=args.fc_hidden_size,
                            stochastic=True).to(device=device)
        dec = FCNDecoderVAE(dim_in=args.dim_z,
                            dim_out=true_dim_x,
                            bn=args.use_batch_norm,
                            drop=args.use_dropout,
                            nl=nl,
                            output_nl=output_nl,
                            hidden_size=args.fc_hidden_size).to(device=device)
    elif args.enc_dec_net == 'cnn':
        enc = FullyConvEncoderVAE(input=args.dim_x[0] + args.frame_stacks,
                                    latent_size=args.dim_z,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    img_dim=str(args.dim_x[1]),
                                    stochastic=True).to(device=device)
        dec = FullyConvDecoderVAE(input=args.dim_x[0] + args.frame_stacks,
                                    latent_size=args.dim_z,
                                    bn=args.use_batch_norm,
                                    drop=args.use_dropout,
                                    nl=nl,
                                    img_dim=str(args.dim_x[1]),
                                    output_nl=output_nl).to(device=device)

    # Dynamics network
    dyn = LinearMixRNN(dim_z=args.dim_z,
                        dim_u=args.dim_u,
                        hidden_size=args.rnn_hidden_size,
                        bidirectional=args.use_bidirectional,
                        net_type=args.rnn_net,
                        K=args.K).to(device=device)
    base_matrices_params = [dyn.A, dyn.B]

    if args.opt == "adam":
        opt_vae = torch.optim.Adam(list(enc.parameters()) + 
                                   list(dec.parameters()), 
                                   lr=args.lr)
        opt_vae_base = torch.optim.Adam(list(enc.parameters()) + 
                                        list(dec.parameters()) + 
                                        base_matrices_params, 
                                        lr=args.lr)
        opt_all = torch.optim.Adam(list(enc.parameters()) + 
                                   list(dec.parameters()) +
                                   list(dyn.parameters()), 
                                   lr=args.lr)
    elif args.opt == "sgd":
        opt_vae = torch.optim.SGD(list(enc.parameters()) + 
                                  list(dec.parameters()), 
                                  lr=args.lr, 
                                  momentum=0.9, 
                                  nesterov=True)
        opt_vae_base = torch.optim.SGD(list(enc.parameters()) + 
                                     list(dec.parameters()) + 
                                     base_matrices_params,
                                     lr=args.lr, 
                                     momentum=0.9, 
                                     nesterov=True)
        opt_all = torch.optim.SGD(list(enc.parameters()) + 
                                  list(dec.parameters()) +
                                  list(dyn.parameters()), 
                                  lr=args.lr, 
                                  momentum=0.9, 
                                  nesterov=True)
    else:
        raise NotImplementedError()

    if args.weight_init == 'custom':
        enc.apply(common_init_weights)
        dec.apply(common_init_weights)
        dyn.apply(common_init_weights)
    
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
                   img_shape=args.dim_x)
    ds_size = len(ds)
    idx = list(range(ds_size))
    split = int(np.floor(args.val_split * ds_size))
    train_idx, val_idx = idx[split:], idx[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(ds,
                              batch_size=args.n_batch,
                              num_workers=args.n_worker,
                              sampler=train_sampler,
                              worker_init_fn=_init_fn)
    val_loader = DataLoader(ds,
                            batch_size=args.n_batch,
                            num_workers=args.n_worker,
                            sampler=valid_sampler,
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
            x_full = data['images'].float().to(device=device) # (n, l, 1, h, w)
            x_full = frame_stack(x_full, frames=args.frame_stacks) # (n, l - frames, 1 * frames, h, w)

            start_idx = np.random.randint(x_full.shape[1] - args.traj_len + 1) # sample random range of traj_len
            end_idx = start_idx + args.traj_len
            x = x_full[:, start_idx:end_idx]
            n = x.shape[0]
            l = x.shape[1]
            x = x.reshape(n * l, *x.shape[2:]) # reshape to (n * l, 1, height, width)
            u = data['actions'][:, start_idx:end_idx].float().to(device=device)

            # Encode & Decode all samples
            z, z_mu, z_logvar = enc(x)
            x_hat = dec(z)
            loss_rec = torch.sum(loss_REC(x_hat, x))

            # Dynamics constraint with KL
            z = z.reshape(n, l, *z.shape[1:])
            z_mu = z_mu.reshape(n, l, *z_mu.shape[1:])
            z_logvar = z_logvar.reshape(n, l, *z_logvar.shape[1:])
            z_var = torch.diag_embed(torch.exp(z_logvar))

            z_t1_hat, z_mu_t1_hat, z_var_t1_hat, _ = dyn(z_t=z[:, :-1], mu_t=z_mu[:, :-1], 
                                                         var_t=z_var[:, :-1], u=u[:, 1:])

            # Initial distribution 
            z_mu_i = torch.zeros(args.dim_z, requires_grad=False, device=device)
            z_var_i = 20.00 * torch.eye(args.dim_z, requires_grad=False, device=device)
            z_mu_i = z_mu_i.repeat(n, 1, 1) # (n, l, dim_z)
            z_var_i = z_var_i.repeat(n, 1, 1, 1) # (n, l, dim_z, dim_z)

            z_mu_hat = torch.cat((z_mu_i, z_mu_t1_hat), 1)
            z_var_hat = torch.cat((z_var_i, z_var_t1_hat), 1)

            loss_kl = torch.sum(kl(mu0=z_mu.reshape(n * l, *z_mu.shape[2:]), 
                                    cov0=z_var.reshape(n * l, *z_var.shape[2:]), 
                                    mu1=z_mu_hat.reshape(n * l, *z_mu_hat.shape[2:]), 
                                    cov1=z_var_hat.reshape(n * l, *z_var_hat.shape[2:])))

            #TODO: Reward prediction
            total_loss = args.lam_rec * loss_rec + args.lam_kl * loss_kl

            total_loss = torch.sum(total_loss) / n
            avg_l.append(total_loss.item())

            # Jointly optimize everything
            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 0.5)
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
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_base
            
            # Train for one epoch        
            avg_train_loss = opt_iter(epoch=epoch, opt=opt)

            # Calculate validtion loss
            if args.val_split > 0:
                with torch.no_grad():
                    avg_val_loss = opt_iter(epoch=epoch)
            else:
                avg_val_loss = 0
            epoch_time = time.time() - tic

            print(f"Epoch {epoch + 1}/{args.n_epoch}: Avg train loss: {avg_train_loss}, Avg val loss: {avg_val_loss}, Time per epoch: {epoch_time}")
            
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
                torch.save(dyn.state_dict(), save_dir + '/dyn.pth')
                torch.save(enc.state_dict(), save_dir + '/enc.pth')
                torch.save(dec.state_dict(), save_dir + '/dec.pth')
            writer.close()

def main():
    args = parse_training_args()
    train(args)

if __name__ == "__main__":
    main()
