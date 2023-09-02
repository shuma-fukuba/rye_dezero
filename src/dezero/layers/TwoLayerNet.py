from dezero import Model
from dezero.core import Parameter
from dezero.functions import sigmoid

from .Layer import Layer
from .Linear import Linear


class TwoLayerNet(Model):
    def __init__(self, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, inputs: Parameter) -> Parameter:
        y = sigmoid(self.l1(inputs))
        y = self.l2(y)
        return y
