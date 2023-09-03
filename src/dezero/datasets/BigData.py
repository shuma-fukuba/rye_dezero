from .Dataset import Dataset
import numpy as np


class BigData(Dataset):
    def __getitem__(self, index: int):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__(self) -> int:
        return 1000000
