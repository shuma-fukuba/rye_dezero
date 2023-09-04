import contextlib
import weakref

import numpy as np

import dezero

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndarray


class Config:
    enable_backprop: bool = True


class Variable:
    __array_priority__ = 200

    def __init__(self, data: array_types, name: str = None) -> None:
        if data is not None:
            if not isinstance(data, array_types):
                raise TypeError(f"{type(data)} is not supported.")

        self.data = data
        self.grad: Variable = None
        self.creator = None
        self.name = name
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def reshape(self, *shape: int):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"

        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func) -> None:
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def to_cpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)


class Function:
    def __init__(self) -> None:
        self.inputs = None
        self.outputs = None
        self.generation = 0

    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        inputs = [as_variable(x) for x in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x: array_types) -> array_types:
        raise NotImplementedError()

    def backward(self, gy: Variable) -> Variable:
        raise NotImplementedError()


class Parameter(Variable):
    pass


class Add(Function):
    def __init__(self) -> None:
        self.x0_shape = None
        self.x1_shape = None

    def forward(self, x0: array_types, x1: array_types) -> array_types:
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Sub(Function):
    def forward(self, x0: array_types, x1: array_types) -> array_types:
        return x0 - x1

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        return gy, -gy


class Mul(Function):
    def forward(self, x0: array_types, x1: array_types) -> array_types:
        y = x0 * x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        return gy * x1, gy * x0


class Neg(Function):
    def forward(self, x: array_types) -> array_types:
        return -x

    def backward(self, gy: Variable) -> Variable:
        return -gy


class Div(Function):
    def forward(self, x0: array_types, x1: array_types) -> array_types:
        y = x0 / x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c: float) -> None:
        super().__init__()
        self.c = c

    def forward(self, x: array_types) -> array_types:
        y = x**self.c
        return y

    def backward(self, gy: Variable) -> Variable:
        (x,) = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def add(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x0, x1)


def radd(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Add()(x1, x0)


def mul(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x0, x1)


def rmul(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Mul()(x1, x0)


def neg(x: Variable) -> Variable:
    return Neg()(x)


def sub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Sub()(x1, x0)


def div(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: Variable) -> Variable:
    x1 = as_array(x1, dezero.cuda.get_array_module(x0.data))
    return Div()(x1, x0)


def pow_(x: Variable, c: float | int) -> Variable:
    return Pow(c)(x)


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


def as_array(x, array_module=np) -> array_types:
    if np.isscalar(x):
        return array_module.array(x)
    return x


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)
