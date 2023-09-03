from typing import Callable

import numpy as np


class Dataset:
    def __init__(
        self,
        train: bool = True,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index: int):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), \
                    self.target_transform(self.label[index])

    def prepare(self):
        pass

    def __len__(self) -> int:
        return len(self.data)
