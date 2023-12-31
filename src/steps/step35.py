import numpy as np

from dezero import Variable
from dezero import functions as F
from dezero.utils import plot_dot_graph

if __name__ == "__main__":
    x = Variable(np.linspace(-7, 7, 200))
    y = F.tanh(x)
    x.name = "x"
    y.name = "y"
    y.backward(create_graph=True)

    iters = 8

    for i in range(iters):
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)

    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
    plot_dot_graph(gx, verbose=False, to_file='tanh.png')
