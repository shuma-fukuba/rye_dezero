from dezero import Variable, utils
from dezero.layers import Layer


class Model(Layer):
    def plot(self, *inputs: Variable, to_file: str = "model.png"):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
