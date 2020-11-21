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

class BAIRPush(object):
    """BAIR robot push datasets. Modified from: https://github.com/edenton/svg"""
    def __init__(self, dir, train=True, seq_len=20, offset_control=True):
        self.root_dir = dir 
        if train:
            data_dir = os.path.join(self.root_dir, "train")
            self.ordered = False
        else:
            data_dir = os.path.join(self.root_dir, "test")
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(data_dir):
            for d2 in os.listdir('%s/%s' % (data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (data_dir, d1, d2))
        self.seq_len = seq_len
        self.d = 0
        self.offset_control = offset_control
          
    def __len__(self):
        return len(self.dirs)

    def get_seq(self, idx):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[idx]
        image_seq = []
        for i in range(self.seq_len):
            fname = f'{d}/{i}.png'
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)
        action_seq = pkl_loader(f'{d}/actions.pkl')[:self.seq_len]
        ee_pos_seq = pkl_loader(f'{d}/ee_pos.pkl')[:self.seq_len]

        # If control indexing is different offset data with dummy variable to match
        if self.offset_control:
            padding = np.zeros((1, action_seq.shape[-1]))
            action_seq = np.concatenate(
                (padding, action_seq[:-1]), 
            )

        sample = {
            'img':image_seq, # (T, 1, res, res) 
            'action': action_seq,
            'gt': ee_pos_seq
        }

        return sample

    def __getitem__(self, index):
        return self.get_seq(index)

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

class ImgCached(data.Dataset):
    """Image Dataset from a single cached tuple file of np.arrays
       (image, action) or (image, action, gtstate). 
       Raw cached images assumed to be of shape (n, l, w, h, c=3).
    """
    def __init__(self, dir, loader=pkl_loader, transform=None, rgb=False):
        """
        Args:
            dir (string): Directory of the cache.
            loader (callable): Function to load a sample given its path.
        """
        self.dir = dir
        self.transform = transform

        print("Loading cache for dataset")
        data = loader(dir) 

        cached_data_raw, self.cached_data_actions, self.cached_data_state = data
  
        print("Formating dataset")
        batch_size = cached_data_raw.shape[0]
        traj_len = cached_data_raw.shape[1]

        cached_data_raw = cached_data_raw.reshape(batch_size, traj_len, 64, 64, 3)

        if rgb:
            self.cached_data = np.transpose(cached_data_raw, (0, 1, 4, 2, 3)) / 255.0
        else:
            self.cached_data = np.expand_dims(rgb2gray(cached_data_raw)[..., 0], axis=2)
        
        if transform is not None:
            for ii in range(batch_size):
                for tt in range(traj_len):
                    self.cached_data[ii, tt, :, :, :] = transform(self.cached_data[ii, tt, :, :, :])

    def __len__(self):
        return self.cached_data.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            sample (dict): 
        """
        assert(idx < self.__len__()), "Index must be lower than dataset size " + str(self.__len__())
        img = self.cached_data[idx] # (T, 1, res, res) 
        a = self.cached_data_actions[idx] # (T, 1)
        gt = self.cached_data_state[idx]

        sample = {'img':img, # (T, 1, res, res) 
                  'action': a,
                  'gt': gt}

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dir Location: {}\n'.format(self.dir)
        tmp = '    Image Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str