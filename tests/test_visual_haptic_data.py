import os, sys
os.sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from datasets import VisualHaptic
from models import HapticNet
from utils import frame_stack

def test_vh():
    dataset = VisualHaptic(
            "/Users/oliver/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_1D_B1F515581A0A478A92AF1C58D4345408.pkl",
            img_shape=(3,64,64)
        )

    ds_size = len(dataset)
    idx = list(range(ds_size))
    split = int(np.floor(0 * ds_size))
    train_idx, val_idx = idx[split:], idx[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        sampler=train_sampler
    )

    net = HapticNet(input_size=6, num_channels=[450, 450, 8])

    for idx, data in enumerate(train_loader):
        img = data["img"].float()
        ft = data["ft"].float()
        ee_pose = data["arm"].float()
        u = data["action"].float()
        # print("img", img.shape)
        # print("ft", ft.shape)
        # print("arm", ee_pose.shape)
        # print("action", u.shape)

        ft = frame_stack(ft)
        ft = ft.reshape(-1, *ft.shape[2:]) 
        out = net(ft)
        break

if __name__=="__main__":
    test_vh()
