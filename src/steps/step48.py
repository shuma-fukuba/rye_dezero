import math

import numpy as np

import dezero.functions as F
from dezero import optimizers
from dezero.datasets import get_spiral
from dezero.models import MLP

if __name__ == "__main__":
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    x, t = get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_iter):
        index = np.random.permutation(data_size)
        sum_loss = 0
        for i in range(max_iter):
            batch_index = index[i * batch_size : (i + 1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.clear_grads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))
