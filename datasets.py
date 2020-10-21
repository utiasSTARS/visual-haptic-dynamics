"""
PyTorch wrappers for datasets.
"""
import torch.utils.data as data
import torch
import numpy as np
import pickle as pkl
from utils import rgb2gray
import os, io

from skimage.io import imread
from PIL import Image

def pkl_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

class BAIRPush(object):
    """BAIR robot push datasets. Modified from: https://github.com/edenton/svg"""
    def __init__(self, dir, train=True, seq_len=20, offset_control=True):
        self.root_dir = dir 
        if train:
            self.data_dir = os.path.join(self.root_dir, "train")
            self.ordered = False
        else:
            self.data_dir = os.path.join(self.root_dir, "test")
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
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
    def __init__(self, dir, loader=pkl_loader, transform=None, rgb=False):
        """
        Args:
            dir (string): Directory of the cache.
            loader (callable): Function to load a sample given its path.
        """
        self.dir = dir

        print("Loading cache for dataset")
        self.data = loader(dir) 
        print("Formating dataset")        
        batch_size = self.data["img"].shape[0]
        traj_len = self.data["img"].shape[1]

        if rgb:
            self.data["img"] = np.transpose(self.data["img"], (0, 1, 4, 2, 3)) / 255.0
        else:
            self.data["img"] = np.expand_dims(rgb2gray(self.data["img"])[..., 0], axis=2)

        self.data["ft"] /= 100.0
        
        if transform is not None:
            for ii in range(batch_size):
                for tt in range(traj_len):
                    self.data["img"][ii, tt, :, :, :] = transform(self.data["img"][ii, tt, :, :, :])

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