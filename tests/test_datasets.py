import os, sys
os.sys.path.insert(0, "..")

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from datasets import ImgCached, VisualHaptic, BAIRPush

def test_bair():
    print("Testing BAIR")
    dataset = BAIRPush(dir="/media/m2-drive/datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl")
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
        print(idx)
        print("img", data["img"].shape)
        print("action", data["action"].shape)
        print("gt", data["gt"].shape)
        break

def test_cached():
    print("Testing Cached (pendulum/reacher)")
    dataset = ImgCached(dir="/media/m2-drive/datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl")
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
        print(idx)
        print("img", data["img"].shape)
        print("action", data["action"].shape)
        print("gt", data["gt"].shape)
        break

def test_vh():
    print("Testing visual haptic")
    dataset = VisualHaptic(
            dir="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_withGT_3D9E4376CF4746EEA20DCD520218038D.pkl",
            rgb=True
        )
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
        print(idx)
        print("img", data["img"].shape)
        print("ft", data["ft"].shape)
        print("arm", data["arm"].shape)
        print("action", data["action"].shape)
        break

if __name__=="__main__":
    # test_cached()
    # test_vh()
    test_bair()