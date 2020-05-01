"""
Compare consistency and speed between manual 
implementation and Torch distribituion package KL divergence.
"""
import os, sys, time
sys.path.append('../')

from losses import KL

def test_KL_single():
    tic = time.time()
    mu1 = torch.Tensor([[1., 2., 3.],
                    [2., 3., 4.]])
    var_1 = torch.Tensor([[1., 1., 1.],
                        [4., 9., 16.]])

	mu2 = torch.Tensor([[1., 3., 4.],
						[2., 3., 4.]])
	var_2 = torch.Tensor([[1., 4., 9.],
						  [4., 9., 16.]])

	p = torch.distributions.Normal(mu1, var_1)
	q = torch.distributions.Normal(mu2, var_2)
	kl_loss_torch = torch.distributions.kl_divergence(p, q)
    toc = time.time() - tic

	print("KL from Torch distribution package: {}, time taken: {}",
            kl_loss_torch, toc)

def test_KL_batch():
    pass

if __name__=="__main__":
  test_KL()
