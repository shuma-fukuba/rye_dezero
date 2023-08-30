import numpy as np

from dezero import Variable


def f(x: Variable) -> Variable:
    y = x ** 4 - 2 * x ** 2
    return y


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)

        x.clear_grad()
        y.backward(create_graph=True)
        gx = x.grad
        x.clear_grad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data
