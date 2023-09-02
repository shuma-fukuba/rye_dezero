from .Optimizer import Optimizer
from dezero.core import Parameter


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter):
        param.data -= self.lr * param.grad.data
