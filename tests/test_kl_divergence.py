"""
Compare consistency and speed between manual 
implementation and Torch distribituion package KL divergence.
"""
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
from losses import kl

def test_KL(bs=1000):
    mu0 = -torch.rand((bs,3))
    var0 = 3*torch.eye(3).repeat(bs,1,1)
    mu1 = torch.rand((bs,3))
    var1 = 2.5*torch.eye(3).repeat(bs,1,1)
    
    # Test Pytorch's KL divergence
    tic = time.time()
    p = torch.distributions.MultivariateNormal(mu0, var0)
    q = torch.distributions.MultivariateNormal(mu1, var1)
    kl_loss_torch = torch.distributions.kl_divergence(p, q)
    toc = time.time() - tic

    print(f"Torch distribution package: {kl_loss_torch}, Time Taken: {toc}")
    
    # Test custom KL
    tic = time.time()
    kl_loss_custom = kl(mu0=mu0, cov0=var0, mu1=mu1, cov1=var1)
    toc = time.time() - tic

    print(f"Custom: {kl_loss_custom}, Time Taken: {toc}")

    mse = torch.sum(kl_loss_torch - kl_loss_custom)**2
    print(f"Mean Squared Error: {mse:.15f}")

if __name__=="__main__":
    test_KL()
