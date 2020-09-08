import os, sys
os.sys.path.insert(0, "..")

import torch
import torch.nn as nn
from networks import (
    CNNEncoder1D,
    CNNDecoder1D
)

if __name__=="__main__":
    test_inp = torch.zeros((32, 15, 12, 32))
    test_inp = test_inp.float().to(device="cuda").reshape(-1, *test_inp.shape[2:])

    haptic_enc = CNNEncoder1D(
        input=12,
        latent_size=16,
        bn=True,
        drop=False,
        nl=nn.ReLU(),
        stochastic=False
    ).to(device="cuda")

    haptic_dec = CNNDecoder1D(
        input=12,
        latent_size=16,
        bn=True,
        drop=False,
        nl=nn.ReLU(),
    ).to(device="cuda")

    print("in", test_inp.shape)
    z = haptic_enc(test_inp)
    print("z", z.shape)
    out = haptic_dec(z)
    print("out", out.shape)