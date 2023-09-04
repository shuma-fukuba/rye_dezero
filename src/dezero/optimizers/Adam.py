import math

import numpy as np

from dezero.core import Parameter

from .Optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self) -> float:
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param: Parameter) -> None:
        # xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            # self.ms[key] = xp.zeros_like(param.data)
            # self.vs[key] = xp.zeros_like(param.data)
            self.ms[key] = np.zeros_like(param.data)
            self.vs[key] = np.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        # param.data -= self.lr * m / (xp.sqrt(v) + eps)
        param.data -= self.lr * m / (np.sqrt(v) + eps)
