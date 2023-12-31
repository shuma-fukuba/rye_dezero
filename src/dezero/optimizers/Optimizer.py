from dezero.core import Parameter
from dezero.models import Model


class Optimizer:
    def __init__(self) -> None:
        self.target: Model = None
        self.hooks = []

    def setup(self, target: Model):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        # パラメータの更新
        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)
