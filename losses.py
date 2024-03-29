"""
Custom losses.
"""
import torch

def kl(mu0, cov0, mu1, cov1, eps=1e-8):
    """
    KL(N0 || N1), Kl divergence between two gaussians N0 ~ N(mu0, cov0) and 
    N1 ~ N(mu1, cov1), where cov0 and cov 1 are specifically diagonal matrices.
    mu0, mu1: (bs, dim)
    cov0, cov1: (bs, dim, dim)
    """
    sumf = lambda x: torch.sum(x, dim=-1)
    prodf = lambda x: torch.prod(x, dim=-1) 

    k = mu0.shape[-1]
    var0 = torch.diagonal(cov0, dim1=-2, dim2=-1) + eps # (bs, dim)
    var1 = torch.diagonal(cov1, dim1=-2, dim2=-1) + eps # (bs, dim)
    ivar1 = 1.0 / var1

    a = sumf(ivar1 * var0) # (bs,)
    b = sumf((mu1 - mu0)**2 * ivar1)
    c = -k
    d = sumf(torch.log(var1)) - sumf(torch.log(var0))
    loss_kl = torch.sum(0.5 * (a+b+c+d))
    return loss_kl

def torch_kl(mu0, cov0, mu1, cov1):
    p = torch.distributions.MultivariateNormal(
        mu0, 
        cov0
    )
    q = torch.distributions.MultivariateNormal(
        mu1, 
        cov1
    )

    loss_kl = torch.sum(
        torch.distributions.kl_divergence(p, q)
    ) 
    return loss_kl