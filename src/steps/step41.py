import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    W = Variable(np.random.randn(3, 4))
    y = F.matmul(x, W)
    y.backward()

    print(x.grad.shape)
    print(W.grad.shape)
