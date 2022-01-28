"""
PyTorch wrappers for datasets.
"""
import torch.utils.data as data
import torch
import numpy as np
import pickle as pkl
import gzip
from utils import rgb2gray
import os, io

from skimage.io import imread
from PIL import Image

def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return test_f.read(2) == b'\x1f\x8b'

def pkl_loader(path):
    """A data loader for pickle files."""
    if is_gz_file(path):
        with gzip.open(path, 'rb') as f:
            data = pkl.load(f)
    else:
        with open(path, 'rb') as f:
            data = pkl.load(f)
    return data

class VisualHaptic(data.Dataset):
    def __init__(self, dir, loader=pkl_loader, img_transform=None, rgb=False, normalize_ft=100):
        """
        Args:
            dir (string): Directory of the cache.
            loader (callable): Function to load a sample given its path.
        """
        self.rgb = rgb
        self.transform = img_transform
        self.loader = loader
        self.extra_dirs = []
        self.normalize_ft = normalize_ft

        print("Loading cache for dataset")
        self.dir = dir
        data = loader(dir) 
        self.original_data_size = data["img"].shape[0]
        self.data = self.format_data(data)

    def format_data(self, data):
        batch_size = data["img"].shape[0]
        traj_len = data["img"].shape[1]

        if self.rgb:
            data["img"] = np.transpose(data["img"], (0, 1, 4, 2, 3)) / 255.0
        else:
            data["img"] = np.expand_dims(rgb2gray(data["img"])[..., 0], axis=2)

        data["ft"] /= self.normalize_ft
        
        if self.transform is not None:
            for ii in range(batch_size):
                for tt in range(traj_len):
                    data["img"][ii, tt, :, :, :] = self.transform(data["img"][ii, tt, :, :, :])
        return data

    def __len__(self):
        return self.data["img"].shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            sample (dict): 
        """
        assert(idx < self.__len__()), "Index must be lower than dataset size " + str(self.__len__())

        sample = {'img': self.data["img"][idx], # (T, 1, res, res) 
                  'ft': self.data["ft"][idx],
                  'arm': self.data["arm"][idx],
                  'action': self.data["action"][idx],
                  'gt': self.data["gt_plate_pos"][idx]}

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dir Location: {}\n'.format(self.dir)
        return fmt_str

    def append_cache(self, dir, format=True):
        """Add new cached "offline" datasets to the original."""
        self.extra_dirs.append(dir)
        data = self.loader(dir) 
        if format:
            data = self.format_data(data)

        if not self.data["config"] == data["config"]:
            print("Warning: adding data with different env configs")
            print("Original: ", self.data["config"])
            print("Appended: ", data["config"])

        for key in data.keys():
            if key == "config":
                continue
            assert key in self.data, "Adding data not in the original dataset."
            self.data[key] = np.concatenate((self.data[key], data[key]))

    def append(self, data, format=True):
        """Add new trajectories to the dataset for "online" training."""
        if format:
            data = self.format_data(data)

        for key in data.keys():
            assert key in self.data, "Adding data not in the original dataset."
            self.data[key] = np.concatenate((self.data[key], data[key]))

    def get_appended_data(self):
        """Return only data which was appended."""
        appended_data = {}
        for key in self.data.keys():
            if key == "config":
                pass
            else:
                appended_data[key] = self.data[key][self.original_data_size:]
        return appended_data