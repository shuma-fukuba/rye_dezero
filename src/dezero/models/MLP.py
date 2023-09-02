from dezero.core import Parameter
from dezero.functions import sigmoid
from dezero.layers import Linear

from .BaseModel import Model


class MLP(Model):
    def __init__(self, fc_output_sizes: list[int], activation=sigmoid) -> None:
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: Parameter) -> Parameter:
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
