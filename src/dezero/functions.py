import numpy as np

from dezero.core import Function, Variable, as_variable


class Sin(Function):
    def forward(self, x: np.ndarray):
        y = np.sin(x)
        return y

    def backward(self, gy: Variable):
        (x,) = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = gy * -sin(x)
        return gx


class Tanh(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


class Reshape(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: Variable):
        return reshape(gy, self.x_shape)


def cos(x: Variable) -> Variable:
    return Cos()(x)


def sin(x: Variable) -> Variable:
    return Sin()(x)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


def reshape(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)
