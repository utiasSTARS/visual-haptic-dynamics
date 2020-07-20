from utils import (set_seed_torch, 
                    common_init_weights, 
                    Normalize,
                    frame_stack)
import numpy as np
import random
from args.parser import parse_vh_training_args
from collections import OrderedDict
from datetime import datetime
from tensorboardX import SummaryWriter
import json
import os, sys, time
import re

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import (FullyConvEncoderVAE,
                        FullyConvDecoderVAE,
                        FCNEncoderVAE,
                        FCNDecoderVAE)
from models import (LinearMixSSM, 
                    LinearSSM, 
                    NonLinearSSM,
                    TCN)
from datasets import VisualHaptic

def train(args):
    print(args)
    set_seed_torch(args.random_seed)
    def _init_fn(worker_id):
        np.random.seed(int(args.random_seed))

    assert 0 <= args.opt_vae_epochs <= args.opt_vae_base_epochs <= args.n_epoch
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # Keeping track of results and hyperparameters
    save_dir = os.path.join(args.storage_base_path, args.comment)
    checkpoint_dir = os.path.join(save_dir, "checkpoints/")

    if not args.debug:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        args.__dict__ = OrderedDict(
            sorted(args.__dict__.items(), key=lambda t: t[0])
        )
        with open(save_dir + '/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        writer = SummaryWriter(logdir=save_dir)

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
    rec_modalities = []

    # Networks
    if args.use_img_enc:
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

    if args.use_haptic_enc:
        haptic_enc = TCN(
            input_size=6,
            num_channels=list(args.tcn_channels)
        ).to(device=device)
        nets["haptic_enc"] = haptic_enc
        z_dim_in += args.dim_z_haptic

    if args.use_arm_enc:
        arm_enc = TCN(
            input_size=6,
            num_channels=list(args.tcn_channels)
        ).to(device=device)
        nets["arm_enc"] = arm_enc
        z_dim_in += args.dim_z_arm

    if args.use_img_dec:
        img_dec = FullyConvDecoderVAE(
            input=args.dim_x[0] * (args.frame_stacks + 1),
            latent_size=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            img_dim=args.dim_x[1],
            output_nl=None if args.use_binary_ce else nn.Sigmoid()
        ).to(device=device)
        nets["img_dec"] = img_dec
        rec_modalities.append("img")

    if args.use_haptic_dec:
        haptic_dec = FCNDecoderVAE(
            dim_in=args.dim_z, 
            dim_out=6 * 32 * (args.frame_stacks + 1), 
            bn=args.use_batch_norm, 
            drop=args.use_dropout, 
            nl=nn.ReLU(), 
            output_nl=None, 
            hidden_size=args.fc_hidden_size
        ).to(device=device)
        nets["haptic_dec"] = haptic_dec
        rec_modalities.append("haptic")

    if args.use_arm_dec:
        arm_dec = FCNDecoderVAE(
            dim_in=args.dim_z, 
            dim_out=6 * 32 * (args.frame_stacks + 1), 
            bn=args.use_batch_norm, 
            drop=args.use_dropout, 
            nl=nn.ReLU(), 
            output_nl=None, 
            hidden_size=args.fc_hidden_size
        ).to(device=device)
        nets["arm_dec"] = arm_dec
        rec_modalities.append("arm")

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
        base_params = [dyn.A, dyn.B]
    elif args.dyn_net == "linearrank1":
        dyn = LinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
        base_params = []
    elif args.dyn_net == "nonlinear":
        dyn = NonLinearSSM(
            dim_z=args.dim_z,
            dim_u=args.dim_u,
            hidden_size=args.rnn_hidden_size,
            bidirectional=args.use_bidirectional,
            net_type=args.rnn_net
        ).to(device=device)
        base_params = []
    else:
        raise NotImplementedError()
    nets["dyn"] = dyn

    enc_params = [list(v.parameters()) for k, v in nets.items() if "enc" in k]
    enc_params = [v for sl in enc_params for v in sl] # remove nested list

    dec_params = [list(v.parameters()) for k, v in nets.items() if "dec" in k]
    dec_params = [v for sl in dec_params for v in sl] # remove nested list

    if args.opt == "adam":
        opt_type = torch.optim.Adam
    elif args.opt == "sgd":
        opt_type = torch.optim.SGD
    else:
        raise NotImplementedError()

    opt_vae = opt_type(
        enc_params + 
        dec_params +
        list(nets["mix"].parameters()),
        lr=args.lr
    )
    opt_vae_base = opt_type(
        enc_params +
        dec_params +
        list(nets["mix"].parameters()) +
        base_params, 
        lr=args.lr
    )
    opt_all = opt_type(
        enc_params +
        dec_params +
        list(nets["mix"].parameters()) +
        list(nets["dyn"].parameters()), 
        lr=args.lr
    )
     
    if args.weight_init == 'custom':
        for k, v in nets.items():
            v.apply(common_init_weights)
    
    # Loss functions
    if args.use_binary_ce:
        loss_REC = nn.BCEWithLogitsLoss(reduction='none').to(device=device)
    else:
        loss_REC = nn.MSELoss(reduction='none').to(device=device)

    # Dataset
    dataset = VisualHaptic(
                  args.dataset,
                  img_shape=args.dim_x
              )

    idx = list(range(len(dataset)))
    split = int(np.floor(args.val_split * len(dataset)))
    train_sampler = SubsetRandomSampler(idx[split:])
    valid_sampler = SubsetRandomSampler(idx[:split])

    train_loader = DataLoader(
        dataset,
        batch_size=args.n_batch,
        num_workers=args.n_worker,
        sampler=train_sampler,
        worker_init_fn=_init_fn
    )
    val_loader = DataLoader(
        dataset,
        batch_size=args.n_batch,
        num_workers=args.n_worker,
        sampler=valid_sampler,
        worker_init_fn=_init_fn
    )

    #XXX: If a checkpoint exists, assumed preempted and resume training
    checkpoint_epochs = 0
    if os.path.exists(checkpoint_dir + "checkpoint.pth"):
        checkpoint = torch.load(checkpoint_dir + "checkpoint.pth")
        for k, v in nets.items():
            v.load_state_dict(checkpoint[k])
        opt_vae.load_state_dict(checkpoint['opt_vae'])
        opt_vae_base.load_state_dict(checkpoint['opt_vae_base'])
        opt_all.load_state_dict(checkpoint['opt_all'])
        checkpoint_epochs = checkpoint['epoch']
        print(f"Resuming training from checkpoint at epoch {checkpoint_epochs}")
        assert (checkpoint_epochs < args.n_epoch), \
            f"""The amount of epochs {args.n_epoch} should be greater 
            than the already trained checkpoint epochs {checkpoint_epochs}"""

    def opt_iter(epoch, opt=None):
        """Single training epoch."""
        if opt:
            for k, v in nets.items():
                v.train()
            loader = train_loader
        else:
            for k, v in nets.items():
                v.eval()
            loader = val_loader
        
        # Keep track of losses
        running_stats = {"total_l": [], "kl_l": []}
        for m in rec_modalities:
            running_stats[f'rec_l_{m}'] = []

        for idx, data in enumerate(loader):
            if idx == args.n_example:
                break
            
            # Load and shape trajectory data
            # XXX: all trajectories have same length
            x = {}
            x['img'] = data['img'].float().to(device=device) # (n, l, c, h, w)
            x['haptic'] = data['ft'].float().to(device=device) # (n, l, f, 6)
            x['arm'] = data['arm'].float().to(device=device) # (n, l, f, 6)

            n = x['img'].shape[0]

            x['img'] = frame_stack(x['img'], frames=args.frame_stacks)

            start_idx = np.random.randint(x['img'].shape[1] - args.traj_len + 1) # sample random range of traj_len
            end_idx = start_idx + args.traj_len
            for k in x:
                x[k] = x[k][:, start_idx:end_idx]

            l = x['img'].shape[1]

            for k in x:
                x[k] = x[k].reshape(-1, *x[k].shape[2:])
            
            u = data['action'][:, (start_idx + args.frame_stacks):(end_idx + args.frame_stacks)].float().to(device=device)

            # Encode
            z_all = []
            if args.use_img_enc:
                z_all.append(nets["img_enc"](x['img']))
            if args.use_haptic_enc:
                z_all.append(nets["haptic_enc"](x['haptic'])[:, -1])
            if args.use_arm_enc:
                z_all.append(nets["arm_enc"](x['arm'])[:, -1])

            # Concatenate modalities
            z_cat = torch.cat(z_all, dim=1)

            z, mu_z, logvar_z = nets["mix"](z_cat)

            # Decode
            loss_rec = {}
            x_hat = {}
            for m in rec_modalities:
                x_hat[f"{m}"] = nets[f"{m}_dec"](z)
                if m in ["haptic", "arm"]:
                    x_hat[f"{m}"] = x_hat[f"{m}"].reshape(*x[f'{m}'].shape)
                    loss_rec[f"loss_rec_{m}"] = (torch.sum(loss_REC(x_hat[f"{m}"], x[f'{m}']))) / n
                else:
                    loss_rec[f"loss_rec_{m}"] = (torch.sum(loss_REC(x_hat[f"{m}"], x[f'{m}']))) / n

            # Dynamics constraint with KL
            z = z.reshape(n, l, *z.shape[1:])
            mu_z = mu_z.reshape(n, l, *mu_z.shape[1:])
            logvar_z = logvar_z.reshape(n, l, *logvar_z.shape[1:])
            var_z = torch.diag_embed(torch.exp(logvar_z))

            _, mu_z_t1_hat, var_z_t1_hat, _ = nets["dyn"](
                z_t=z[:, :-1].transpose(1,0), 
                mu_t=mu_z[:, :-1].transpose(1,0), 
                var_t=var_z[:, :-1].transpose(1,0), 
                u=u[:, 1:].transpose(1,0)
            )
            mu_z_t1_hat = mu_z_t1_hat.transpose(1,0)
            var_z_t1_hat = var_z_t1_hat.transpose(1,0)

            # Initial distribution 
            mu_z_i = torch.zeros(
                args.dim_z, 
                requires_grad=False, 
                device=device
            ).repeat(n, 1, 1)

            var_z_i = 20.00 * torch.eye(
                args.dim_z, 
                requires_grad=False, 
                device=device
            ).repeat(n, 1, 1, 1) 

            mu_z_hat = torch.cat((mu_z_i, mu_z_t1_hat), 1)
            var_z_hat = torch.cat((var_z_i, var_z_t1_hat), 1)

            p = torch.distributions.MultivariateNormal(
                mu_z.reshape(-1, *mu_z.shape[2:]), 
                var_z.reshape(-1, *var_z.shape[2:])
            )
            q = torch.distributions.MultivariateNormal(
                mu_z_hat.reshape(-1, *mu_z_hat.shape[2:]), 
                var_z_hat.reshape(-1, *var_z_hat.shape[2:])
            )
            loss_kl = torch.sum(torch.distributions.kl_divergence(p, q)) / n
            running_stats['kl_l'].append(loss_kl.item())

            loss_rec_total = 0
            for m in rec_modalities:
                loss_rec_total += loss_rec[f"loss_rec_{m}"]
                running_stats[f'rec_l_{m}'].append(loss_rec[f"loss_rec_{m}"].item())

            total_loss = args.lam_rec * loss_rec_total + \
                args.lam_kl * loss_kl
            running_stats['total_l'].append(
                loss_rec_total.item() +
                loss_kl.item()
            )

            # Jointly optimize everything
            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                for k, v in nets.items():
                    torch.nn.utils.clip_grad_norm_(v.parameters(), 0.5)
                opt.step()

        # Summary stats from epoch
        summary_stats = {f'avg_{k}':sum(v)/len(v) for k, v in running_stats.items()}
        n_images = 16 # random sample of images to visualize reconstruction quality
        rng = random.randint(0, x["img"].shape[0] - n_images)
        summary_stats['og_imgs'] = x["img"][rng:(rng + n_images), np.newaxis, -1].detach().cpu()
        summary_stats['rec_imgs'] = x_hat["img"][rng:(rng + n_images), np.newaxis, -1].detach().cpu()

        return summary_stats

    # Training loop
    opt = opt_vae
    try:
        for epoch in range(1 + checkpoint_epochs, args.n_epoch + 1):
            tic = time.time()
            if epoch >= args.opt_vae_base_epochs:
                opt = opt_all
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_base
            
            # Train for one epoch        
            summary_train = opt_iter(epoch=epoch, opt=opt)

            # Calculate validtion loss
            if args.val_split > 0:
                with torch.no_grad():
                    summary_val = opt_iter(epoch=epoch)
            epoch_time = time.time() - tic

            print((f"Epoch {epoch}/{args.n_epoch}: " 
                f"Avg train loss: {summary_train['avg_total_l']}, " 
                f"Avg val loss: {summary_val['avg_total_l'] if args.val_split > 0 else 'N/A'}, "
                f"Time per epoch: {epoch_time}"))
            
            if not args.debug:
                # Tensorboard
                for m in rec_modalities:
                    writer.add_scalar(f"loss/{m}/train", summary_train[f'avg_rec_l_{m}'], epoch)
                for loss in ['total', 'kl']:
                    writer.add_scalar(f"loss/{loss}/train", summary_train[f'avg_{loss}_l'], epoch)
                writer.add_images(f'reconstructed_images/{args.comment}/train/original', summary_train['og_imgs'])
                writer.add_images(f'reconstructed_images/{args.comment}/train/reconstructed', summary_train['rec_imgs'])

                if args.val_split > 0:
                    for m in rec_modalities:
                        writer.add_scalar(f"loss/{m}/val", summary_val[f'avg_rec_l_{m}'], epoch)
                    for loss in ['total', 'kl']:
                        writer.add_scalar(f"loss/{loss}/val", summary_val[f'avg_{loss}_l'], epoch)
                    writer.add_images(f'reconstructed_images/{args.comment}/val/original', summary_val['og_imgs'])
                    writer.add_images(f'reconstructed_images/{args.comment}/val/reconstructed', summary_val['rec_imgs'])

                # Save model at intermittent checkpoints 
                if epoch % args.n_checkpoint_epoch == 0:
                    torch.save(
                        {**{k: v.state_dict() for k, v in nets.items()},
                        'opt_all': opt_all.state_dict(),
                        'opt_vae': opt_vae.state_dict(),
                        'opt_vae_base': opt_vae_base.state_dict(),
                        'epoch': epoch}, 
                        checkpoint_dir + "checkpoint.pth"
                    )
    finally:
        if not args.debug:
            if not np.isnan(summary_train['avg_total_l']):
                # Save models
                for k, v in nets.items():
                    torch.save(v.state_dict(), save_dir + f"/{k}.pth")
            writer.close()

def main():
    args = parse_vh_training_args()
    train(args)

if __name__ == "__main__":
    main()
