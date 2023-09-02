import weakref
from typing import Any

from dezero.core import Parameter


class Layer:
    def __init__(self) -> None:
        self._params: set[str] = set()
        self.inputs = []
        self.outputs = []

    def __setattr__(self, __name: str, __value: Any) -> None:
        if isinstance(__value, (Parameter, Layer)):
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

    def params(self) -> Parameter:
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()
