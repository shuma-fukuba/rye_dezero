import matplotlib.pyplot as plt
import numpy as np

from dezero import DataLoader, no_grad
from dezero.datasets import MNIST
from dezero.functions import accuracy, relu, softmax_cross_entropy
from dezero.models import MLP
from dezero.optimizers import Adam


def f(x: np.ndarray):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


if __name__ == "__main__":
    train_set = MNIST(train=True, transform=f)
    test_set = MNIST(train=False, transform=f)

    max_epoch = 5
    batch_size = 100
    hidden_size = 1000

    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, hidden_size, 10), activation=relu)
    optimizer = Adam().setup(model)

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
        print(f"epoch: {float(epoch + 1)}")
        print(
            "train loss: {:.4f}, accuracy: {:.4f}".format(
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
            "test loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(test_set), sum_acc / len(test_set)
            )
        )
