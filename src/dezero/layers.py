import weakref
from typing import Any

import numpy as np
import dezero.functions as F
from dezero.core import Parameter


class Layer:
    def __init__(self) -> None:
        self._params: set[Parameter] = set()
        self.inputs = []
        self.outputs = []

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, Parameter):
            self._params.add(__name)
        super().__setattr__(__name, __value)

    def __call__(self, *inputs: Parameter) -> Parameter | list[Parameter]:
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs: Parameter):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            yield self.__dict__[name]

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()


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

        return F.linear(x, self.W, self.b)
