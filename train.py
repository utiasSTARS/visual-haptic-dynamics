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
                        FCNDecoderVAE,
                        CNNEncoder1D,
                        CNNDecoder1D)
from models import (LinearMixSSM, 
                    LinearSSM, 
                    NonLinearSSM)
from datasets import VisualHaptic
from losses import torch_kl

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
            latent_size=args.dim_z_img,
            bn=args.use_batch_norm,
            drop=args.use_dropout,
            nl=nl,
            img_dim=args.dim_x[1],
            stochastic=False
        ).to(device=device)
        nets["context_img_enc"] = context_img_enc
        z_dim_in += args.dim_z_context

        # context_img_dec = FullyConvDecoderVAE(
        #     input=args.dim_x[0] * (args.frame_stacks + 1),
        #     latent_size=dim_z_rec,
        #     bn=args.use_batch_norm,
        #     drop=args.use_dropout,
        #     nl=nl,
        #     img_dim=args.dim_x[1],
        #     output_nl=None if args.use_binary_ce else nn.Sigmoid()
        # ).to(device=device)
        # nets["img_dec"] = img_dec

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
        running_stats = {"total_l": [], "kl_l": [], "rec_l_img": []}

        for idx, data in enumerate(loader):
            if idx == args.n_example:
                break
            
            # Load and shape trajectory data
            x = {}
            x['img'] = data['img'].float().to(device=device) # (n, l, c, h, w)
            x['img'] = frame_stack(x['img'], frames=args.frame_stacks)
            ep_len = x['img'].shape[1]

            u = data['action'].float().to(device=device)
            u = u[:, args.frame_stacks:]

            if args.context_modality != "none":
                if args.context_modality == "joint": 
                    x["context"] = torch.cat((data['ft'], data['arm']), dim=-1) # (n, l, f, 12)
                elif args.context_modality == "ft" or "arm": 
                    x["context"] = data[args.context_modality]
                x["context"] = x["context"].float().to(device=device) # (n, l, f, 6)
                x["context"] = x["context"].transpose(-1, -2)
                x["context"] = x["context"][:, args.frame_stacks:]

            # Randomly and uniformly sample
            ll = np.random.randint(ep_len-1)
            x_ll = {}

            for k in x:
                x_ll[k] = x[k][:, ll:]
            u_ll = u[:, ll:]
            n, l = x_ll['img'].shape[0], x_ll['img'].shape[1]

            x_ll['context_img'] = x_ll["img"][:, 0].unsqueeze(1).repeat(1, l, 1, 1, 1)
            x_ll = {k:v.reshape(-1, *v.shape[2:]) for k, v in x_ll.items()}

            # 1. Encoding            
            z_all_enc = []
            z_all_enc.append(nets["img_enc"](x_ll['img']))
            
            if args.context_modality != "none":
                z_context = nets["context_enc"](x_ll["context"])
                z_all_enc.append(z_context)
            if args.use_context_img:
                z_img_context = nets["context_img_enc"](x_ll['context_img'])
                z_all_enc.append(z_img_context)

            # Concatenate modalities and mix
            z_cat_enc = torch.cat(z_all_enc, dim=1)
            z, mu_z, logvar_z = nets["mix"](z_cat_enc)
            var_z = torch.diag_embed(torch.exp(logvar_z))

            # Group sample, mean, covariance
            q_z = {"z": z, "mu": mu_z, "cov": var_z}
            
            # 2. Reconstruction
            z_all_dec = []
            z_all_dec.append(q_z["z"])

            if args.context_modality != "none":
                z_all_dec.append(z_context)
            if args.use_context_img:
                z_all_dec.append(z_img_context)
            z_cat_dec = torch.cat(z_all_dec, dim=1)

            loss_rec = 0
            x_hat_img = nets["img_dec"](z_cat_dec)
            loss_rec_img = (torch.sum(
                loss_REC(x_hat_img, x_ll['img'])
            )) / n
            running_stats['rec_l_img'].append(loss_rec_img.item())
            loss_rec += loss_rec_img

            # 3. Dynamics constraint with KL
            loss_kl = 0

            # Unflatten and transpose seq_len and batch for convenience
            q_z = {k:v.reshape(n, l, *v.shape[1:]).transpose(1,0) for k, v in q_z.items()}
            u_ll = u_ll.transpose(1,0)

            # Initial distribution
            mu_z_i = torch.zeros(
                args.dim_z, 
                requires_grad=False, 
                device=device
            ).repeat(1, n, 1)

            var_z_i = 20.00 * torch.eye(
                args.dim_z, 
                requires_grad=False, 
                device=device
            ).repeat(1, n, 1, 1)

            loss_kl += torch_kl(
                mu0=q_z["mu"][0:1],
                cov0=q_z["cov"][0:1],
                mu1=mu_z_i,
                cov1=var_z_i
            ) / n

            # Prior transition distributions
            z_t1_hat, mu_z_t1_hat, var_z_t1_hat, (h_t, _) = nets["dyn"](
                z_t=q_z["z"][:-1], 
                mu_t=q_z["mu"][:-1], 
                var_t=q_z["cov"][:-1], 
                u=u_ll[1:],
                h_0=None,
                return_all_hidden=True
            )
            p_z = {"z": z_t1_hat, "mu": mu_z_t1_hat, "cov": var_z_t1_hat}

            loss_kl += torch_kl(
                mu0=q_z["mu"][1:],
                cov0=q_z["cov"][1:],
                mu1=p_z["mu"],
                cov1=p_z["cov"]
            ) / n

            # Original length before calculating n-step predictions
            length = p_z["mu"].shape[0]

            # N-step transition distributions
            if epoch > args.opt_n_step_pred_epochs:
                # New references for convenience
                p_z_nstep = p_z
                q_z_nstep = {k:v[1:] for k, v in q_z.items()}
                u_nstep = u_ll[1:]

                for ii in range(min(args.n_step_pred - 1, length - 1)):
                    p_z_nstep = {k:v[:-1] for k, v in p_z_nstep.items()}
                    h_t = h_t[:-1]
                    u_nstep = u_nstep[1:]
                    q_z_nstep = {k:v[1:] for k, v in q_z_nstep.items()}

                    l_nstep = p_z_nstep["z"].shape[0]
                    n_nstep = p_z_nstep["z"].shape[1]

                    p_z_nstep = {k:v.reshape(-1, *v.shape[2:]) for k, v in p_z_nstep.items()}
                    u_nstep = u_nstep.reshape(-1, *u_nstep.shape[2:])
                    h_t = h_t.reshape(-1, *h_t.shape[2:])

                    z_nstep_t1, mu_z_nstep_t1, var_z_nstep_t1, (h_t1, _) = nets["dyn"](
                        z_t=p_z_nstep["z"], 
                        mu_t=p_z_nstep["mu"], 
                        var_t=p_z_nstep["cov"], 
                        u=u_nstep,
                        h_0=h_t,
                        return_all_hidden=True,
                        single=True
                    )

                    p_z_nstep.update({
                        "z": z_nstep_t1, 
                        "mu": mu_z_nstep_t1, 
                        "cov": var_z_nstep_t1
                    })
                    h_t = h_t1

                    p_z_nstep = {k:v.reshape(l_nstep, n_nstep, *v.shape[1:]) for k, v in p_z_nstep.items()}
                    u_nstep = u_nstep.reshape(l_nstep, n_nstep, *u_nstep.shape[1:])
                    h_t = h_t.reshape(l_nstep, n_nstep, *h_t.shape[1:])

                    loss_kl += torch_kl(
                        mu0=q_z_nstep["mu"],
                        cov0=q_z_nstep["cov"],
                        mu1=p_z_nstep["mu"],
                        cov1=p_z_nstep["cov"]
                    ) / n

            running_stats['kl_l'].append(
                loss_kl.item()
            )

            # Jointly optimize everything
            total_loss = args.lam_rec * loss_rec + \
                args.lam_kl * loss_kl
                
            running_stats['total_l'].append(
                loss_rec.item() +
                loss_kl.item()
            )

            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                for k, v in nets.items():
                    torch.nn.utils.clip_grad_norm_(v.parameters(), 0.5)
                opt.step()

        # Summary stats from epoch
        summary_stats = {f'avg_{k}':sum(v)/len(v) for k, v in running_stats.items()}

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
                writer.add_scalar(f"loss/img/train", summary_train['avg_rec_l_img'], epoch)
                for loss in ['total', 'kl']:
                    writer.add_scalar(f"loss/{loss}/train", summary_train[f'avg_{loss}_l'], epoch)

                if args.val_split > 0:
                    writer.add_scalar(f"loss/img/val", summary_val['avg_rec_l_img'], epoch)
                    for loss in ['total', 'kl']:
                        writer.add_scalar(f"loss/{loss}/val", summary_val[f'avg_{loss}_l'], epoch)

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
