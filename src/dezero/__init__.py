is_simple_core = False

if is_simple_core:
    from dezero.core_simple import as_array  # F401: noqa
    from dezero.core_simple import (
        Function,
        Variable,
        add,
        as_variable,
        div,
        mul,
        neg,
        no_grad,
        rdiv,
        rsub,
        sub,
        using_config,
    )
else:
    from dezero.core import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        using_config,
        add,
        mul,
        neg,
        sub,
        rsub,
        div,
        rdiv,
        pow_,
    )

from dezero.functions import get_item, matmul, max_, min_
from dezero.models import Model

from . import datasets
from .dataloaders import DataLoader


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow_
    Variable.__getitem__ = get_item

    Variable.matmul = matmul
    Variable.dot = matmul
    Variable.max = max_
    Variable.min = min_


setup_variable()
