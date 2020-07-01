"""
PyTorch wrappers for datasets.
"""
import torch.utils.data as data
import torch
import numpy as np
import pickle as pkl
from utils import rgb2gray

def pkl_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

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

        self.data["img"] = np.expand_dims(rgb2gray(self.data["img"])[..., 0], axis=2)

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
                  'action': self.data["action"][idx]}

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