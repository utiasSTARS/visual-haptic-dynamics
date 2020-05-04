"""
Compare consistency and speed between manual 
implementation and Torch distribituion package KL divergence.
"""
import os, sys, time
sys.path.append('../')

import torch
from pprint import pprint
from losses import kl

def test_KL_single():
    mu0 = torch.rand((6,3))
    var0 = torch.eye(3).repeat(6,1,1)
    mu1 = torch.rand((6,3))
    var1 = torch.eye(3).repeat(6,1,1)
    
    # Test Pytorch's KL divergence
    tic = time.time()
    p = torch.distributions.MultivariateNormal(mu0, var0)
    q = torch.distributions.MultivariateNormal(mu1, var1)
    kl_loss_torch = torch.distributions.kl_divergence(p, q)
    toc = time.time() - tic

    pprint({"Torch distribution package": kl_loss_torch, 
            "time taken": toc})
    
    # Test custom KL
    tic = time.time()
    kl_loss_custom = kl(mu0=mu0, cov0=var0, mu1=mu1, cov1=var1)
    toc = time.time() - tic

    pprint({"Torch distribution package": kl_loss_custom, 
            "time taken": toc})

def test_KL_batch():
    pass

if __name__=="__main__":
    test_KL_single()
