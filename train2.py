from utils2 import (
    set_seed_torch, 
    common_init_weights, 
    Normalize,
    frame_stack,
    load_vh_models
)
import numpy as np
import random
from args.parser import parse_vh_training_args
from collections import OrderedDict
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import json
import os, sys, time
import re
import pickle as pkl
import bisect

import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from networks import (
    FullyConvEncoderVAE,
    FullyConvDecoderVAE,
    FCNEncoderVAE,
    FCNDecoderVAE,
    CNNEncoder1D,
    RNNEncoder
)
from models import (
    LinearMixSSM, 
    NonLinearSSM,
    ProductOfExperts
)
from datasets import VisualHaptic, ImgCached
from losses import torch_kl

def setup_opt_iter(args):
    # Loss functions
    if args.use_binary_ce:
        loss_REC = nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_REC = nn.MSELoss(reduction='none')
    
    if args.context_modality != "none":
        poe = ProductOfExperts()

    def opt_iter(loader, nets, device, opt=None, n_step=1):
        """Single training epoch."""
        if opt:
            for k, v in nets.items():
                v.train()
        else:
            for k, v in nets.items():
                v.eval()
        
        # Keep track of losses
        running_stats = {"total_l": [], "kl_l": [], "img_rec_l": []}
        if args.reconstruct_context and args.context_modality != "none":
            running_stats["context_rec_l"] = []

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
                if args.use_context_frame_stack:
                    x['context'] = frame_stack(x['context'], frames=args.frame_stacks)
                x["context"] = x["context"].transpose(-1, -2)
                if not args.use_context_frame_stack:
                    x["context"] = x["context"][:, args.frame_stacks:]

            # Randomly and uniformly sample?
            # ll = np.random.randint(ep_len-1)

            # Train from index 0 all the time
            ll = 0
            x_ll = {}

            for k in x:
                x_ll[k] = x[k][:, ll:]
            u_ll = u[:, ll:]
            n, l = x_ll['img'].shape[0], x_ll['img'].shape[1]
            x_ll = {k:v.reshape(-1, *v.shape[2:]) for k, v in x_ll.items()}

            # 1. Encoding q(z) distribution
            z_img, mu_z_img, logvar_z_img = nets["img_enc"](x_ll['img'])
            if args.context_modality != "none":
                _, mu_z_context, logvar_z_context = nets["context_enc"](x_ll["context"])
                # Mix modalities with product of experts
                mu_z_obs_enc, logvar_z_obs_enc  = poe(
                    mu=torch.cat((
                        mu_z_img.unsqueeze(1), 
                        mu_z_context.unsqueeze(1)
                    ), axis=1),
                    logvar=torch.cat((
                        logvar_z_img.unsqueeze(1), 
                        logvar_z_context.unsqueeze(1)
                    ), axis=1), 
                )

                std_z_obs_enc = torch.exp(logvar_z_obs_enc / 2.0)
                eps = torch.randn_like(std_z_obs_enc)
                z_obs_enc = mu_z_obs_enc + eps * std_z_obs_enc
            else:
                # If only image is used
                z_obs_enc, mu_z_obs_enc, logvar_z_obs_enc = \
                    z_img, mu_z_img, logvar_z_img

            if args.context == "none":
                # If only CNN is used for recognition, process in batch instead of roll out
                var_z_obs_enc = torch.diag_embed(torch.exp(logvar_z_obs_enc))
                # Group sample, mean, covariance
                q_z = {"z": z_obs_enc, "mu": mu_z_obs_enc, "cov": var_z_obs_enc}
            elif args.context == "ssm" or "rssm":
                q_z = {
                    "z": torch.empty((n, l, args.dim_z), device=device),
                    "mu": torch.empty((n, l, args.dim_z), device=device),
                    "cov": torch.empty((n, l, args.dim_z, args.dim_z), device=device)
                }

                z_cat_enc = z_obs_enc.reshape(
                    n, l, *z_obs_enc.shape[1:]
                ).transpose(1,0)

                z_0 =torch.zeros(
                    args.dim_z, 
                    requires_grad=False, 
                    device=device
                ).repeat(n, 1)

                if args.context == "rssm":
                    h_0 = None

                # Roll out
                for ii in range(l):
                    if args.context == "ssm":
                        inp = z_0
                    elif args.context == "rssm":
                        z_0 = z_0.unsqueeze(0)
                        out, h = nets["rssm_enc"](z_0, h_0)
                        inp = out[0]
                        h_0 = h

                    z, mu_z, logvar_z = nets["mix"](
                        torch.cat((inp, z_cat_enc[ii]), axis=-1)
                    )
                    var_z = torch.diag_embed(torch.exp(logvar_z))
                    q_z["z"][:, ii] = z 
                    q_z["mu"][:, ii] = mu_z
                    q_z["cov"][:, ii] = var_z
                    z_0 = z
                    
                q_z = {k:v.reshape(-1, *v.shape[2:]) for k, v in q_z.items()}

            # 2. Reconstruction
            x_hat_img = nets["img_dec"](q_z["z"])
            loss_rec_img = (torch.sum(
                loss_REC(x_hat_img, x_ll['img'])
            )) / n
            running_stats['img_rec_l'].append(loss_rec_img.item())

            if args.context_modality != "none" and args.reconstruct_context:
                x_hat_context = nets["context_dec"](q_z["z"])
                loss_rec_context = (torch.sum(
                    loss_REC(x_hat_context, x_ll['context'])
                )) / n
                running_stats['context_rec_l'].append(loss_rec_context.item())

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
                mu0=q_z["mu"],
                cov0=q_z["cov"],
                mu1=torch.cat((mu_z_i, p_z["mu"]), axis=0),
                cov1=torch.cat((var_z_i, p_z["cov"]), axis=0)
            ) / n

            # Original length before calculating n-step predictions
            length = p_z["mu"].shape[0]

            # N-step transition distributions
            #XXX: This doesn't work with LSTM
            if n_step > 1:
                # New references for convenience
                p_z_nstep = p_z
                q_z_nstep = {k:v[1:] for k, v in q_z.items()}
                u_nstep = u_ll[1:]

                for ii in range(min(n_step - 1, length - 1)):
                    p_z_nstep = {k:v[:-1] for k, v in p_z_nstep.items()}
                    h_t = h_t[:-1]
                    u_nstep = u_nstep[1:]
                    q_z_nstep = {k:v[1:] for k, v in q_z_nstep.items()}

                    l_nstep = p_z_nstep["z"].shape[0]
                    n_nstep = p_z_nstep["z"].shape[1]

                    p_z_nstep = {k:v.reshape(-1, *v.shape[2:]) for k, v in p_z_nstep.items()}
                    u_nstep = u_nstep.reshape(-1, *u_nstep.shape[2:])
                    h_t = h_t.reshape(-1, *h_t.shape[2:]).unsqueeze(0)

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
                    h_t = h_t1[0]

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

            if args.context_modality != "none" and args.reconstruct_context:
                running_stats['total_l'].append(
                    loss_rec_img.item() +
                    loss_rec_context.item() +
                    loss_kl.item()
                )
                # Jointly optimize everything
                total_loss = \
                    args.lam_rec * loss_rec_img + \
                    args.lam_rec * loss_rec_context + \
                    args.lam_kl * loss_kl 
            else:
                running_stats['total_l'].append(
                    loss_rec_img.item() +
                    loss_kl.item()
                )
                # Jointly optimize everything
                total_loss = \
                    args.lam_rec * loss_rec_img + \
                    args.lam_kl * loss_kl 

            if opt:
                opt.zero_grad()
                total_loss.backward()
                # clip for stable RNN training
                for k, v in nets.items():
                    torch.nn.utils.clip_grad_norm_(v.parameters(), 0.50)
                opt.step()

        # Summary stats from epoch
        summary_stats = {f'avg_{k}':sum(v)/len(v) for k, v in running_stats.items()}

        return summary_stats
    return opt_iter

def train(args):
    print(args)

    assert 0 <= args.opt_vae_epochs <= args.opt_vae_base_epochs <= args.n_epoch

    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    set_seed_torch(args.random_seed)
    def _init_fn(worker_id):
        np.random.seed(int(args.random_seed))

    device = torch.device(args.device)

    # Keeping track of results and hyperparameters
    save_dir = os.path.join(args.storage_base_path, args.comment)

    slurm_id = os.environ.get('SLURM_JOB_ID')
    if slurm_id is not None:
        user = os.environ.get('USER')
        checkpoint_dir = f"/checkpoint/{user}/{slurm_id}"
    else:
        checkpoint_dir = os.path.join(save_dir, "checkpoints/")

    if not args.debug:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        args.__dict__ = OrderedDict(
            sorted(args.__dict__.items(), key=lambda t: t[0])
        )
        with open(save_dir + '/hyperparameters.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        writer = SummaryWriter(log_dir=save_dir)
        tb_data = []

    # Setup network models
    nets = load_vh_models(args, device=device)

    if args.weight_init == 'custom':
        for k, v in nets.items():
            v.apply(common_init_weights)

    # Setup optimizers
    if args.opt == "adam":
        opt_type = torch.optim.Adam
    elif args.opt == "sgd":
        opt_type = torch.optim.SGD
    elif args.opt == "rmsprop":
        opt_type = torch.optim.RMSprop
    else:
        raise NotImplementedError()
    
    if args.dyn_net == "linearmix":
        base_params = [nets["dyn"].A, nets["dyn"].B]
    elif args.dyn_net == "nonlinear":
        base_params = []
    else:
        raise NotImplementedError()

    if args.context == "rssm":
        extra_params = list(nets["mix"].parameters())
    else:
        extra_params = []

    enc_params = [list(v.parameters()) for k, v in nets.items() if "enc" in k]
    enc_params = [v for sl in enc_params for v in sl] # remove nested list

    dec_params = [list(v.parameters()) for k, v in nets.items() if "dec" in k]
    dec_params = [v for sl in dec_params for v in sl] # remove nested list

    opt_vae = opt_type(
        enc_params + 
        dec_params +
        extra_params,
        lr=args.lr
    )
    opt_vae_base = opt_type(
        enc_params +
        dec_params +
        extra_params +
        base_params, 
        lr=args.lr
    )
    opt_all = opt_type(
        enc_params +
        dec_params +
        extra_params +
        list(nets["dyn"].parameters()), 
        lr=args.lr
    )

    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_all)

    # Setup dataset
    if args.task == "push64vh":
        dataset = VisualHaptic(
            args.dataset[0],
            rgb=args.dim_x[0] == 3,
            normalize_ft = args.ft_normalization
        )
    elif args.task == "pendulum64":
        dataset = ImgCached(
            args.dataset[0],
            rgb=args.dim_x[0] == 3,
        )

    # Append any extra datasets
    for extra_dataset in args.dataset[1:]:
        dataset.append_cache(extra_dataset)

    dataset_idx = list(range(len(dataset)))
    random.shuffle(dataset_idx)

    split = int(np.floor(args.val_split * len(dataset)))

    train_sampler = SubsetRandomSampler(dataset_idx[split:])
    valid_sampler = SubsetRandomSampler(dataset_idx[:split])

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
    
    opt_iter = setup_opt_iter(args)
    n_step_lookup = list(args.opt_n_step_pred_epochs)

    # Training loop
    try:
        opt = opt_vae
        for epoch in range(checkpoint_epochs + 1, args.n_epoch + 1):
            tic = time.time()
            if epoch >= args.opt_vae_base_epochs:
                opt = opt_all
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_base
            n_step_pred = bisect.bisect_left(n_step_lookup, epoch) + 1

            # Train for one epoch        
            summary_train = opt_iter(
                loader=train_loader, 
                nets=nets, 
                device=device,
                opt=opt,
                n_step=n_step_pred
            )

            # Calculate validtion loss
            if args.val_split > 0:
                with torch.no_grad():
                    summary_val = opt_iter(
                        loader=val_loader,
                        nets=nets,
                        device=device,
                        n_step=n_step_pred
                    )
                if args.use_scheduler and epoch >= args.opt_vae_base_epochs:
                    scheduler.step(summary_val['avg_total_l'])

            epoch_time = time.time() - tic

            print((
                f"Epoch {epoch}/{args.n_epoch}, Time per epoch: {epoch_time}: "
                f"\n[Train] "
                f"Total: {summary_train['avg_total_l']}, "
                f"Image rec: {summary_train['avg_img_rec_l']}, "
                f"Context rec: {summary_train['avg_context_rec_l'] if (args.reconstruct_context and args.context_modality != 'none') else 'N/A'}, "
                f"KL: {summary_train['avg_kl_l']}"
                f"\n[Val] "
                f"Total: : {summary_val['avg_total_l'] if (args.val_split > 0) else 'N/A'}, "
                f"Image rec: {summary_val['avg_img_rec_l'] if (args.val_split > 0) else 'N/A'}, "
                f"Context rec: {summary_val['avg_context_rec_l'] if (args.val_split > 0 and args.reconstruct_context and args.context_modality != 'none') else 'N/A'}, "
                f"KL: {summary_val['avg_kl_l'] if (args.val_split > 0) else 'N/A'}"
            ))

            if not args.debug:
                # Temporarily store tensorboard data
                for k, v in summary_train.items():
                    tb_data.append((f"train/{k}", v, epoch))

                if args.val_split > 0:
                    for k, v in summary_val.items():
                        tb_data.append((f"val/{k}", v, epoch))

                if epoch % args.n_checkpoint_epoch == 0:
                    # Write tensorboard data
                    for data in tb_data:
                        writer.add_scalar(data[0], data[1], data[2])
                    tb_data = []

                    # Save model at intermittent checkpoints 
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
            # Save models
            for k, v in nets.items():
                torch.save(v.state_dict(), save_dir + f"/{k}.pth")
            if args.val_split > 0:
                with open(save_dir + "/val_idx.pkl", "wb") as f:
                    pkl.dump(dataset_idx[split:], f)
            writer.close()

def main():
    args = parse_vh_training_args()
    train(args)

if __name__ == "__main__":
    main()
