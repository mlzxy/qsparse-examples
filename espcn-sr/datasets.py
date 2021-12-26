import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file, is_rgb=False):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.is_rgb = is_rgb

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            l, r = f["lr"][idx] / 255.0, f["hr"][idx] / 255.0
            if self.is_rgb:
                l, r = np.transpose(l, (2, 0, 1)), np.transpose(r, (2, 0, 1))
            else:
                l, r = np.expand_dims(l, 0), np.expand_dims(r, 0)
            return l, r

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["lr"])


class EvalDataset(Dataset):
    def __init__(self, h5_file, is_rgb=False):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file
        self.is_rgb = is_rgb

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            l, r = f["lr"][str(idx)][:, :] / 255.0, f["hr"][str(idx)][:, :] / 255.0
            if self.is_rgb:
                l, r = np.transpose(l, (2, 0, 1)), np.transpose(r, (2, 0, 1))
            else:
                l, r = np.expand_dims(l, 0), np.expand_dims(r, 0)
            return l, r

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["lr"])
