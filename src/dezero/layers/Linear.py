import numpy as np

from dezero.core import Parameter
from dezero.functions import linear
from .Layer import Layer


class Linear(Layer):
    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype: any = np.float32,
        in_size: int = None,
    ) -> None:
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")

        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x: Parameter) -> Parameter:
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        return linear(x, self.W, self.b)
