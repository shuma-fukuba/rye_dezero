import numpy as np

from dezero import Variable, as_variable
from dezero.functions import exp, sum_
from dezero.models import MLP


def softmax1d(x: Variable) -> Variable:
    x = as_variable(x)
    y = exp(x)
    sum_y = sum_(y)
    return y / sum_y


if __name__ == "__main__":
    model = MLP((10, 3))

    x = Variable(np.array([[0.2, -0.4]]))
    y = model(x)
    p = softmax1d(y)
    print(y)
    print(p)
 