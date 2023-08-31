import numpy as np

from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = x.reshape((2, 3))
    y = x.reshape(2, 3)
    print(y)
