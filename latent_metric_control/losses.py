"""
Custom losses.
"""

def KL(mu0, cov0, mu1, cov1):
    """
    KL(N0 || N1), Kl divergence between two gaussians N0 ~ N(mu0, cov0) and 
    N1 ~ N(mu1, cov1), where cov0 and cov 1 are diagonal matrices.
    mu0, mu1: (bs, dim)
    cov0, cov1: (bs, dim, dim)
    """
    sum_ = lambda x: torch.sum(x, dim=-1)

    k = mu0.shape[1]
    var0 = torch.diagonal(cov0, dim1=-2, dim2=-1) # (bs, dim)
    var1 = torch.diagonal(cov1, dim1=-2, dim2=-1) # (bs, dim)
    ivar0 = 1 / var0
    ivar1 = 1 / var1

    a = sum_(ivar1 * var0) # (bs,)
    b = sum_((mu1 - mu0)**2 * ivar1)
    c = -k
    d = log(sum_(ivar1) / sum_(ivar0))

    return 0.5*(a+b+c+d)