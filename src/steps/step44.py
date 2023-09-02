import numpy as np

import dezero.functions as F
import dezero.layers as L
from dezero import Variable

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    l1 = L.Linear(10)  # 出力サイズ設定
    l2 = L.Linear(1)

    def predict(x):
        y = l1(x)
        y = F.sigmoid(y)
        y = l2(y)
        return y

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        l1.clear_grads()
        l2.clear_grads()
        loss.backward()

        for l in (l1, l2):
            for p in l.params():
                p.data -= lr * p.grad.data
        if i % 1000 == 0:
            print(loss) 
