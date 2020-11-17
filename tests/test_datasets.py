import os, sys
os.sys.path.insert(0, "..")
import matplotlib.pyplot as plt
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import gzip
import pickle as pkl

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch

from datasets import ImgCached, VisualHaptic, BAIRPush

def test_bair():
    print("Testing BAIR")
    dataset = BAIRPush(dir=os.path.join(parent_dir, "experiments/data/datasets/processed_data"))
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
    dataset = ImgCached(
        dir="/media/m2-drive/datasets/pendulum-srl-sim/pendulum64_total_2048_traj_16_repeat_2_with_angle_train.pkl",
        rgb=True
    )
    ds_size = len(dataset)
    idx = list(range(ds_size))
    sampler = SubsetRandomSampler(idx)
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        sampler=sampler
    )

    for idx, data in enumerate(train_loader):
        print(idx)
        print("img", data["img"].shape, "min: ", torch.min(data["img"]), "max: ", torch.max(data["img"]))
        print("action", data["action"].shape)
        print("gt", data["gt"].shape)
        break

def test_vh():
    print("Testing visual haptic")
    dataset = VisualHaptic(
            dir="/home/olimoyo/visual-haptic-dynamics/experiments/data/datasets/visual_haptic_2D_len16_withGT_3D9E4376CF4746EEA20DCD520218038D.pkl",
            rgb=False
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
        print("img", data["img"].shape, "min: ", torch.min(data["img"]), "max: ", torch.max(data["img"]))
        print("ft", data["ft"].shape)
        print("arm", data["arm"].shape)
        print("action", data["action"].shape)
        break

def test_mit_push():
    print("Testing MIT push")

    def load_zipped_pickle(filename):
        with gzip.open(filename, 'rb') as f:
            loaded_object = pkl.load(f)
            return loaded_object

    dataset = VisualHaptic(
            dir=os.path.join(parent_dir, "experiments/data/datasets/mit_push/min-tr2.5_min-rot0.5_len48.pkl"),
            rgb=False,
            loader=load_zipped_pickle,
            normalize_ft=1.0
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
        print("img", data["img"].shape, "min: ", torch.min(data["img"]), "max: ", torch.max(data["img"]))
        print("ft", data["ft"].shape)
        print("arm", data["arm"].shape)
        print("action", data["action"].shape)
        break

if __name__=="__main__":
    # test_cached()
    # test_vh()
    # test_bair()
    test_mit_push()