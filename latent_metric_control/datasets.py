"""
PyTorch wrappers for datasets.
"""

def pickle_loader(path):
    """A data loader for pickle files."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class ImgCached(data.Dataset):
    """Image Dataset from a single cached tuple file of np.arrays
       (image, action) or (image, action, gtstate). 
       Images assumed to be of shape (n, t, w, h, c).
    """
    def __init__(self, dir, loader=pickle_loader, transform=None, img_shape=(1,64,64)):
        """
        Args:
            dir (string): Directory of the cache.
            loader (callable): Function to load a sample given its path.
        """
        self.dir = dir
        self.transform = transform

        print("Loading cache for dataset")
        data = loader(dir) 

        if len(data) == 2:
            cached_data_raw, self.cached_data_actions = data
        elif len(data) == 3:
            cached_data_raw, self.cached_data_actions, self.cached_data_state = data
        else:
            raise NotImplementedError()

        print("Formating dataset")
        n = cached_data_raw.shape[0]
        t = cached_data_raw.shape[1]
        c = cached_data_raw.shape[4]

        cached_data_raw = cached_data_raw.reshape(n, t, img_shape[1], img_shape[2], c)
        cached_data_raw = np.moveaxis(cached_data_raw, -1, 2)

        self.cached_data = torch.zeros(n, t, img.shape[0], img_shape[1], img_shape[2])
        for ii in range(n):
            for tt in range(t):
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

        sample = {'images':img, # (T, 1, res, res) 
                  'actions': a}

        return sample

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Dir Location: {}\n'.format(self.dir)
        tmp = '    Image Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str