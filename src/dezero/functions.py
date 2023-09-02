import numpy as np

from dezero import utils
from dezero.core import Function, Variable, as_variable


class Exp(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.exp(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()  # weakref
        gx = gy * y
        return gx


def exp(x: Variable) -> Variable:
    return Exp()(x)


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


class Square(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x**2

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        gx = 2 * x * gy
        return gx


class Reshape(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def __init__(self, axes: int = None):
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x.transpose(self.axes)
        return y

    def backward(self, gy: Variable) -> Variable:
        if self.axes is None:
            return transpose(gy)
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


class Sum(Function):
    def __init__(self, axis: int, keepdims: bool) -> None:
        self.axis = axis
        self.keepdims = keepdims
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


class BroadcastTo(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape: tuple[int]) -> None:
        self.shape = shape
        self.x_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


class MatMul(Function):
    def forward(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


class MeanSquaredError(Function):
    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        diff = x0 - x1
        gy = broadcast_to(gy, diff.shape)
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


class Linear(Function):
    def forward(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


class Sigmoid(Function):
    def forward(self, x: np.ndarray) -> np.ndarray:
        # y = 1 / (1 + xp.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def cos(x: Variable) -> Variable:
    return Cos()(x)


def sin(x: Variable) -> Variable:
    return Sin()(x)


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


def square(x: Variable) -> Variable:
    return Square()(x)


def reshape(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x: Variable, axes: int = None) -> Variable:
    return Transpose(axes)(x)


def sum_(x: Variable, axis: int = None, keepdims: bool = False) -> Variable:
    return Sum(axis=axis, keepdims=keepdims)(x)


def broadcast_to(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x: Variable, shape: tuple[int]) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def matmul(x: Variable, W: Variable) -> Variable:
    return MatMul()(x, W)


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    return MeanSquaredError()(x0, x1)


def mean_squared_error_simple(x0: Variable, x1: Variable) -> Variable:
    diff = x0 - x1
    return sum_(diff**2) / len(diff)


def linear_simple(x: Variable, W: Variable, b: Variable = None) -> Variable:
    x, W = as_variable(x), as_variable(W)
    t = matmul(x, W)
    if b is None:
        return t

    y = t + b
    t.data = None
    return y


def sigmoid_simple(x: Variable) -> Variable:
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


def linear(x: Variable, W: Variable, b: Variable = None) -> Variable:
    return Linear()(x, W, b)


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)
