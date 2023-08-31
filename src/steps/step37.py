import numpy as np
from dezero.core import Variable
from dezero import functions as F

if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
    t = x + c
    y = F.sum(t)
