from utils import (
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
    LinearSSM, 
    NonLinearSSM
)
from datasets import VisualHaptic
from losses import torch_kl

def setup_opt_iter(args):
    # Loss functions
    if args.use_binary_ce:
        loss_REC = nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_REC = nn.MSELoss(reduction='none')

    def opt_iter(epoch, loader, nets, device, opt=None):
        """Single training epoch."""
        if opt:
            for k, v in nets.items():
                v.train()
        else:
            for k, v in nets.items():
                v.eval()
        
        # Keep track of losses
        running_stats = {"total_l": [], "kl_l": [], "img_rec_l": []}

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

            if args.context not in ["initial_image", "goal_image"]:
                x_ll['target_img'] = x_ll['img']

            if args.context in ["initial_latent_state", "initial_image"]:
                context_img = x_ll["img"][:, 0]
            elif args.context in ["goal_latent_state", "goal_image"]:
                #XXX: Assume last image as goal
                context_img = x_ll["img"][:, -1]

            x_ll = {k:v.reshape(-1, *v.shape[2:]) for k, v in x_ll.items()}

            # 1. Encoding
            if args.context in ["initial_image", "goal_image"]:
                context_img_rep = context_img.unsqueeze(1).repeat(1, l, 1, 1, 1)
                context_img_rep = context_img_rep.reshape(-1, *context_img_rep.shape[2:])
                # Concatenate images
                x_ll['target_img'] = x_ll['img'] - context_img_rep
                x_ll['img'] = torch.cat((x_ll['img'], context_img_rep), dim=1)
                
            z_all_enc = []
            z_img = nets["img_enc"](x_ll['img'])
            z_all_enc.append(z_img)

            if args.context_modality != "none":
                z_context = nets["context_enc"](x_ll["context"])
                z_all_enc.append(z_context)

            if args.context in ["initial_latent_state", "goal_latent_state"]:
                z_img_context = nets["context_img_enc"](context_img)
                # Repeat context for the whole trajectory
                z_img_context = z_img_context.unsqueeze(1).repeat(1, l, 1)
                z_img_context = z_img_context.reshape(-1, z_img_context.shape[-1])
                z_all_enc.append(z_img_context)
            elif args.context in ["all_past_states"]:
                # Use all but current state
                context_input = z_img.reshape(n, l, *z_img.shape[1:])[:, :-1].transpose(1,0)
                z_img_context, _ = nets["context_img_rnn_enc"](context_input)
                pad = torch.zeros((1, *z_img_context.shape[1:])).float().to(device=device)
                z_img_context = torch.cat((pad, z_img_context), dim=0)
                z_img_context = z_img_context.transpose(1, 0)
                z_img_context = z_img_context.reshape(-1, *z_img_context.shape[2:])
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

            if args.context in ["initial_latent_state", "goal_latent_state", "all_past_states"]:
                z_all_dec.append(z_img_context)

            # Concatenate modalities and decode
            z_cat_dec = torch.cat(z_all_dec, dim=1)
            x_hat_img = nets["img_dec"](z_cat_dec)
            loss_rec_img = (torch.sum(
                loss_REC(x_hat_img, x_ll['target_img'])
            )) / (n)

            running_stats['img_rec_l'].append(loss_rec_img.item())

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
            ) / (n)

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

            running_stats['total_l'].append(
                loss_rec_img.item() +
                loss_kl.item()
            )

            # Jointly optimize everything
            total_loss = args.lam_rec * loss_rec_img + \
                args.lam_kl * loss_kl 

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
    return opt_iter

def train(args):
    print(args)

    assert 0 <= args.opt_vae_epochs <= args.opt_vae_base_epochs <= args.n_epoch

    set_seed_torch(args.random_seed)
    def _init_fn(worker_id):
        np.random.seed(int(args.random_seed))

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
        writer = SummaryWriter(log_dir=save_dir)

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
    else:
        raise NotImplementedError()
    
    if args.dyn_net == "linearmix":
        base_params = [nets["dyn"].A, nets["dyn"].B]
    elif args.dyn_net == "linearrank1":
        base_params = []
    elif args.dyn_net == "nonlinear":
        base_params = []
    else:
        raise NotImplementedError()

    enc_params = [list(v.parameters()) for k, v in nets.items() if "enc" in k]
    enc_params = [v for sl in enc_params for v in sl] # remove nested list

    dec_params = [list(v.parameters()) for k, v in nets.items() if "dec" in k]
    dec_params = [v for sl in dec_params for v in sl] # remove nested list

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

    # Setup dataset
    dataset = VisualHaptic(
        args.dataset,
        rgb=args.dim_x[0] == 3
    )

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
    
    # Training loop
    try:
        opt = opt_vae
        for epoch in range(1 + checkpoint_epochs, args.n_epoch + 1):
            tic = time.time()
            if epoch >= args.opt_vae_base_epochs:
                opt = opt_all
            elif epoch >= args.opt_vae_epochs:
                opt = opt_vae_base
            
            # Train for one epoch        
            summary_train = opt_iter(
                epoch=epoch, 
                loader=train_loader, 
                nets=nets, 
                device=device,
                opt=opt
            )

            # Calculate validtion loss
            if args.val_split > 0:
                with torch.no_grad():
                    summary_val = opt_iter(
                        epoch=epoch, 
                        loader=val_loader,
                        nets=nets,
                        device=device
                    )

            epoch_time = time.time() - tic

            print((f"Epoch {epoch}/{args.n_epoch}: " 
                f"Avg train loss: {summary_train['avg_total_l']}, "
                f"Avg val loss: {summary_val['avg_total_l'] if args.val_split > 0 else 'N/A'}, "
                f"Time per epoch: {epoch_time}"))
            
            if not args.debug:
                # Tensorboard 
                for k, v in summary_train.items():
                    writer.add_scalar(f"loss/{k}/train", v, epoch)

                if args.val_split > 0:
                    for k, v in summary_val.items():
                        writer.add_scalar(f"loss/{k}/train", v, epoch)

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
