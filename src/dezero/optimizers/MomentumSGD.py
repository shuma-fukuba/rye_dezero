import numpy as np

from dezero.core import Parameter

from .Optimizer import Optimizer


class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param: Parameter) -> None:
        v_key = id(param)
        if v_key not in self.vs:
            self.vs[v_key] = np.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
