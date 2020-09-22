import os, sys
os.sys.path.insert(0, "..")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from datasets import BAIRPush

def test_bair():
    dataset = BAIRPush(data_root="/Users/oliver/visual-haptic-dynamics/experiments/data/datasets")
    ds_size = len(dataset)
    idx = list(range(ds_size))
    sampler = SubsetRandomSampler(idx)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        sampler=sampler
    )

    for idx, data in enumerate(train_loader):
        print(idx, data.shape)

if __name__=="__main__":
    test_bair()