from os.path import join
import numpy as np
import torch
import gzip

class PancreasDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, lod=9):
        assert(lod < 10)
        
        self.data_dir = data_dir
        self.data = {}
        self.current_lod = lod

        for level in range(lod + 1):
            x = self.load(level=level, prefix='scans', dtype=np.float16, factor=1.0)
            t = self.load(level=level, prefix='labels', dtype=np.uint8, factor=(1.0 / 255.0))
            self.data['lod_{}'.format(level)] = {'x': x, 't': t}

    def load(self, level, prefix, dtype, factor):
        resolution = 2 ** (level)
        path = join(self.data_dir, '_data_{}_res_{}.gz'.format(prefix, resolution))
        with gzip.open(path, 'rb') as f:
            data = f.read()
        data = np.frombuffer(data, dtype=dtype)
        samples = int(data.shape[0] / resolution ** 2)
        data = data.reshape((samples, resolution, resolution))
        data = np.expand_dims((data * factor).astype(np.float32), axis=1)
        return data

    def set_lod(self, lod):
        assert(lod < 10)
        self.current_lod = lod

    def __len__(self):
        return len(self.data['lod_0']['x'])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return {'x': self.data['lod_{}'.format(self.current_lod)]['x'][idx],
                't': self.data['lod_{}'.format(self.current_lod)]['t'][idx]}