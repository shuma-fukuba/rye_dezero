is_simple_core = False

if is_simple_core:
    from dezero.core_simple import (
        Function,
        Variable,
        as_array,  # F401: noqa
        as_variable,
        no_grad,
        setup_variable,
        using_config,
    )
    setup_variable()
else:
    from dezero.core import (
        Function,
        Variable,
        as_array,
        as_variable,
        no_grad,
        setup_variable,
        using_config,
    )

    setup_variable()


from dezero.models import Model
