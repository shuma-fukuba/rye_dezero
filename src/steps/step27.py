import math

import numpy as np

from dezero import Function
from dezero.core_simple import Variable





def my_sin(x: Variable, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == "__main__":
    x = Variable(np.array(np.pi / 4))
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
