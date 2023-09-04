import numpy as np

from dezero import DataLoader, no_grad
from dezero.datasets import Spiral
from dezero.functions import accuracy, softmax_cross_entropy
from dezero.models import MLP
from dezero.optimizers import SGD

if __name__ == "__main__":
    batch_size = 30
    hidden_size = 10
    max_epoch = 300
    lr = 1.0

    train_set = Spiral(train=True)
    test_set = Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size)

    model = MLP((hidden_size, 10))
    optimizer = SGD(lr).setup(model)

    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)
            model.clear_grads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        print(f"epoch: {epoch+1}")
        print(
            "test loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(train_set), sum_acc / len(train_set)
            )
        )

        sum_loss, sum_acc = 0, 0

        with no_grad():
            for x, t in test_loader:
                y = model(x)
                loss = softmax_cross_entropy(y, t)
                acc = accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)
        print(
            "train loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(test_set), sum_acc / len(test_set)
            )
        )
