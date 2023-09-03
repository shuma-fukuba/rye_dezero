import math

import numpy as np

import dezero
from dezero.functions import softmax_cross_entropy
from dezero.models import MLP
from dezero.optimizers import SGD

if __name__ == "__main__":
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral()
    model = MLP((hidden_size, 10))
    optimizer = SGD(lr).setup(model)

    data_size = len(train_set)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_epoch):
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # ミニバッチの取り出し
            batch_index = index[i * batch_size : (i + 1) * batch_size]
            batch = [train_set[i] for i in batch_index]
            batch_x = np.array([example[0] for example in batch])
            batch_t = np.array([example[1] for example in batch])

            y = model(batch_x)
            loss = softmax_cross_entropy(y, batch_t)
            model.clear_grads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        avg_loss = sum_loss / data_size
        print("epoch %d loss %.2f" % (epoch + 1, avg_loss))
