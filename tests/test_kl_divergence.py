"""
Compare consistency and speed between manual 
implementation and Torch distribituion package KL divergence.
"""
import os, sys, time
os.sys.path.insert(0, "..")

import torch
from losses import kl, torch_kl

def test_KL(bs=32, n=16):
    mu0 = torch.ones((16,bs,n))
    var0 = 1.0*torch.eye(n).repeat(16,bs,1,1)
    mu1 = torch.ones((16,bs,n))
    var1 = 2.0*torch.eye(n).repeat(16,bs,1,1)

    # Test Pytorch's KL divergence
    tic = time.time()
    kl_loss_torch = torch_kl(mu0=mu0, cov0=var0, mu1=mu1, cov1=var1)
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
