"""
PyTorch wrappers for datasets.
"""
import torch.utils.data as data
import torch
import numpy as np
import pickle as pkl
from utils import rgb2gray
import os, io
from scipy.misc import imresize, imread
from PIL import Image

def pkl_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

class BAIRPush(object):
    """BAIR robot push datasets. Taken from: https://github.com/edenton/svg"""
    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root 
        if train:
            self.data_dir = '%s/processed_data/train' % self.root_dir
            self.ordered = False
        else:
            self.data_dir = '%s/processed_data/test' % self.root_dir
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir('%s/%s' % (self.data_dir, d1)):
                self.dirs.append('%s/%s/%s' % (self.data_dir, d1, d2))
        self.seq_len = seq_len
        self.image_size = image_size 
        self.seed_is_set = False # multi threaded loading
        self.d = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
          
    def __len__(self):
        return 10000

    def get_seq(self):
        if self.ordered:
            d = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d+=1
        else:
            d = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        for i in range(self.seq_len):
            fname = '%s/%d.png' % (d, i)
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()

class VisualHaptic(data.Dataset):
    def __init__(self, dir, loader=pkl_loader, img_shape=(1,64,64)):
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

        if img_shape[0] == 1:
            self.data["img"] = np.expand_dims(rgb2gray(self.data["img"])[..., 0], axis=2)
        elif img_shape[0] == 3:
            self.data["img"] = np.transpose(self.data["img"], (0, 1, 4, 2, 3)) / 255.0

        self.data["ft"] /= 100.0
        
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
                  'gt_plate_pos': self.data["gt_plate_pos"][idx]}

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
    def __init__(self, dir, loader=pkl_loader, transform=None, img_shape=(1,64,64)):
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

        cached_data_raw = cached_data_raw.reshape(batch_size, traj_len, img_shape[1], img_shape[2], 3)

        self.cached_data = torch.zeros(batch_size, traj_len, img_shape[0], img_shape[1], img_shape[2])

        for ii in range(batch_size):
            for tt in range(traj_len):
                self.cached_data[ii, tt, :, :, :] = transform(cached_data_raw[ii, tt, :, :, :])
            
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

        sample = {'img':img, # (T, 1, res, res) 
                  'action': a}

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dir Location: {}\n'.format(self.dir)
        tmp = '    Image Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str