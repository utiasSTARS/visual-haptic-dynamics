from utils import (set_seed_torch, 
                    common_init_weights, 
                    Normalize,
                    frame_stack)
import numpy as np
import random
from args.parser import parse_img_training_args
from collections import OrderedDict
from datetime import datetime
from tensorboardX import SummaryWriter
import json
import os, sys, time

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
                    NonLinearSSM)
from datasets import VisualHaptic, ImgCached

set_seed_torch(3)
def _init_fn(worker_id):
    np.random.seed(int(3))

def train(args):
    print(args)
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
    output_nl = None if args.use_binary_ce else nn.Sigmoid()

    # Encoder and decoder
    if args.enc_dec_net == 'fcn':
        true_dim_x = (
            args.dim_x[0] * (args.frame_stacks + 1), 
            args.dim_x[1], 
            args.dim_x[2]
        )
        enc = FCNEncoderVAE(
            dim_in=int(np.product(true_dim_x)),
            dim_out=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            hidden_size=args.fc_hidden_size,
            stochastic=True
        ).to(device=device)
        dec = FCNDecoderVAE(
            dim_in=args.dim_z,
            dim_out=true_dim_x,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            output_nl=output_nl,
            hidden_size=args.fc_hidden_size
        ).to(device=device)
    elif args.enc_dec_net == 'cnn':
        enc = FullyConvEncoderVAE(
            input=args.dim_x[0] * (args.frame_stacks + 1),
            latent_size=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            img_dim=args.dim_x[1],
            stochastic=True
        ).to(device=device)
        dec = FullyConvDecoderVAE(
            input=args.dim_x[0] * (args.frame_stacks + 1),
            latent_size=args.dim_z,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            img_dim=args.dim_x[1],
            output_nl=output_nl
        ).to(device=device)

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

    if args.opt == "adam":
        opt_vae = torch.optim.Adam(
            list(enc.parameters()) + 
            list(dec.parameters()), 
            lr=args.lr
        )
        opt_vae_base = torch.optim.Adam(
            list(enc.parameters()) + 
            list(dec.parameters()) + 
            base_params, 
            lr=args.lr
        )
        opt_all = torch.optim.Adam(
            list(enc.parameters()) + 
            list(dec.parameters()) +
            list(dyn.parameters()), 
            lr=args.lr
        )
    elif args.opt == "sgd":
        opt_vae = torch.optim.SGD(
            list(enc.parameters()) + 
            list(dec.parameters()), 
            lr=args.lr, 
            momentum=0.9, 
            nesterov=True
        )
        opt_vae_base = torch.optim.SGD(
            list(enc.parameters()) + 
            list(dec.parameters()) + 
            base_params,
            lr=args.lr, 
            momentum=0.9, 
            nesterov=True
        )
        opt_all = torch.optim.SGD(
            list(enc.parameters()) + 
            list(dec.parameters()) +
            list(dyn.parameters()), 
            lr=args.lr, 
            momentum=0.9, 
            nesterov=True
        )
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

    # Dataset
    if args.task == "pendulum64":
        transform = tv.transforms.Compose([
            tv.transforms.ToPILImage(),
            tv.transforms.Grayscale(num_output_channels=1),
            tv.transforms.ToTensor(),
            Normalize(mean=0.27, var=1.0 - 0.27) # 64x64
        ])
        dataset = ImgCached(
                    args.dataset,
                    transform=transform,
                    img_shape=args.dim_x
                    )
    elif args.task == "push64":
        transform = None
        dataset = VisualHaptic(
                    args.dataset,
                    img_shape=args.dim_x
                    )

    ds_size = len(dataset)
    idx = list(range(ds_size))
    split = int(np.floor(args.val_split * ds_size))
    train_idx, val_idx = idx[split:], idx[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
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
        dyn.load_state_dict(checkpoint['dyn'])
        enc.load_state_dict(checkpoint['enc'])
        dec.load_state_dict(checkpoint['dec'])
        opt_all.load_state_dict(checkpoint['opt_all'])
        opt_vae.load_state_dict(checkpoint['opt_vae'])
        opt_vae_base.load_state_dict(checkpoint['opt_vae_base'])
        checkpoint_epochs = checkpoint['epoch']
        print(f"Resuming training from checkpoint at epoch {checkpoint_epochs}")
        assert (checkpoint_epochs < args.n_epoch), \
            f"""The amount of epochs {args.n_epoch} should be greater 
            than the already trained checkpoint epochs {checkpoint_epochs}"""

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
        
        running_stats = {"total_l": [], "kl_l": [], "rec_l": []}

        for idx, data in enumerate(loader):
            if idx == args.n_example:
                break
            
            # Load and shape trajectory data
            # XXX: all trajectories have same length
            x_full = data['img'].float().to(device=device) # (n, l, 1, h, w)
            x_full = frame_stack(x_full, frames=args.frame_stacks) # (n, l - frames, 1 + frames, h, w)
            start_idx = np.random.randint(x_full.shape[1] - args.traj_len + 1) # sample random range of traj_len
            end_idx = start_idx + args.traj_len
            x = x_full[:, start_idx:end_idx]
            n, l = x.shape[0], x.shape[1]
            x = x.reshape(-1, *x.shape[2:]) # reshape to (-1, 1 * fs, height, width)
            u = data['action'][:, (start_idx + args.frame_stacks):(end_idx + args.frame_stacks)].float().to(device=device)
            
            # Encode & Decode all samples
            z, mu_z, logvar_z = enc(x)

            x_hat = dec(z)
            loss_rec = (torch.sum(loss_REC(x_hat, x))) / n
    
            # Dynamics constraint with KL
            z = z.reshape(n, l, *z.shape[1:])
            mu_z = mu_z.reshape(n, l, *mu_z.shape[1:])
            logvar_z = logvar_z.reshape(n, l, *logvar_z.shape[1:])
            var_z = torch.diag_embed(torch.exp(logvar_z))

            _, mu_z_t1_hat, var_z_t1_hat, _ = \
                dyn(z_t=z[:, :-1], mu_t=mu_z[:, :-1], var_t=var_z[:, :-1], u=u[:, 1:])

            # Initial distribution 
            mu_z_i = torch.zeros(args.dim_z, requires_grad=False, device=device)
            var_z_i = 20.00 * torch.eye(args.dim_z, requires_grad=False, device=device)
            mu_z_i = mu_z_i.repeat(n, 1, 1) # (n, l, dim_z)
            var_z_i = var_z_i.repeat(n, 1, 1, 1) # (n, l, dim_z, dim_z)

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

            total_loss = args.lam_rec * loss_rec + args.lam_kl * loss_kl
            running_stats['total_l'].append(loss_rec.item() + loss_kl.item())
            running_stats['rec_l'].append(loss_rec.item())
            running_stats['kl_l'].append(loss_kl.item())

            # Jointly optimize everything
            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(enc.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), 0.5)
                opt.step()

        # Summary stats from epoch
        summary_stats = {f'avg_{key}':sum(stats)/len(stats) 
            for (key, stats) in running_stats.items()}
        n_images = 16 # random sample of images to visualize reconstruction quality
        rng = random.randint(0, x.shape[0] - n_images)
        x_plt = x[rng:(rng + n_images), np.newaxis, -1].detach().cpu()
        x_hat_plt = x_hat[rng:(rng + n_images), np.newaxis, -1].detach().cpu()
        summary_stats['og_imgs'] = x_plt
        summary_stats['rec_imgs'] = x_hat_plt

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
            
            # Tensorboard
            if not args.debug:
                for loss in ['total', 'kl', 'rec']:
                    writer.add_scalar(f"loss/{loss}/train", summary_train[f'avg_{loss}_l'], epoch)
                writer.add_images(f'reconstructed_images/{args.comment}/train/original', summary_train['og_imgs'])
                writer.add_images(f'reconstructed_images/{args.comment}/train/reconstructed', summary_train['rec_imgs'])

                if args.val_split > 0:
                    for loss in ['total', 'kl', 'rec']:
                        writer.add_scalar(f"loss/{loss}/val", summary_val[f'avg_{loss}_l'], epoch)
                    writer.add_images(f'reconstructed_images/{args.comment}/val/original', summary_val['og_imgs'])
                    writer.add_images(f'reconstructed_images/{args.comment}/val/reconstructed', summary_val['rec_imgs'])

                # Save model at intermittent checkpoints 
                if epoch % args.n_checkpoint_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'dyn': dyn.state_dict(),
                        'enc': enc.state_dict(),
                        'dec': dec.state_dict(),
                        'opt_all': opt_all.state_dict(),
                        'opt_vae': opt_vae.state_dict(),
                        'opt_vae_base': opt_vae_base.state_dict()}, 
                        checkpoint_dir + "checkpoint.pth"
                    )

    finally:
        if not args.debug:
            if not np.isnan(summary_train['avg_total_l']):
                # Save models
                torch.save(dyn.state_dict(), save_dir + '/dyn.pth')
                torch.save(enc.state_dict(), save_dir + '/enc.pth')
                torch.save(dec.state_dict(), save_dir + '/dec.pth')
            writer.close()

def main():
    args = parse_img_training_args()
    train(args)

if __name__ == "__main__":
    main()
