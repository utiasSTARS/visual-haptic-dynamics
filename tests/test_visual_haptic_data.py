import os, sys
os.sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datasets import VisualHaptic
from models import HapticNet

def test_vh(bs=1000):
    dataset = VisualHaptic(
            "/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_1D_B1F515581A0A478A92AF1C58D4345408.pkl",
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

    for idx, data in enumerate(train_loader):
        img = data["img"]
        ft = data["ft"]
        ee_pose = data["arm"]
        u = data["action"]
        print("img", img.shape)
        print("ft", ft.shape)
        print("arm", ee_pose.shape)
        print("action", u.shape)
        plt.imshow(img[0,0,0])
        plt.show()

        break
if __name__=="__main__":
    test_vh()
