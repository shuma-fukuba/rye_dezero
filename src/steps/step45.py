import numpy as np

from dezero import Model, Variable
from dezero.functions import mean_squared_error
from dezero.layers.TwoLayerNet import TwoLayerNet

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    model = TwoLayerNet(hidden_size, 1)

    for i in range(max_iter):
        y_pred = model(x)
        loss = mean_squared_error(y, y_pred)

        model.clear_grads()

        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss)
