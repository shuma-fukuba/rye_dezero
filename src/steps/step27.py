import numpy as np

from dezero import Function
from dezero.core_simple import Variable


class Sin(Function):
    def forward(self, x: np.ndarray):
        y = np.sin(x)
        return y

    def backward(self, gy: np.ndarray):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gy


def sin(x: Variable) -> Variable:
    return Sin()(x)


if __name__ == "__main__":
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
