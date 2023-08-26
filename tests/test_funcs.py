from dezero import square, exp, Variable
import numpy as np

def test_square_backward():
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    assert x.grad == expected
