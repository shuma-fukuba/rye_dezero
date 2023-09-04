import time
import dezero
import dezero.functions as F
from dezero.optimizers import SGD
from dezero.datasets import MNIST
from dezero.dataloaders import DataLoader
from dezero.models import MLP


if __name__ == "__main__":
    max_epoch = 5
    batch_size = 100

    train_set = MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)

    model = MLP((1000, 10))
    optimizer = SGD().setup(model)

    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    for epoch in range(max_epoch):
        start = time.time()
        sum_loss = 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.clear_grads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        elapsed_time = time.time() - start
        print(
            "epoch: {}, loss: {:.4f}, time: {:.4f}[sec]".format(
                epoch + 1, sum_loss / len(train_set), elapsed_time
            )
        )
