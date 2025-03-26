import inspect
from collections.abc import Callable
from functools import wraps

__all__ = ["deprecated_positional", "rank_zero_only"]


def deprecated_positional(func: Callable) -> Callable:
    """Remind users to use keyword arguments instead of positional arguments.

    Args:
    ----
        func: The function to wrap.

    """
    from src.utils._logging_utils import get_logger

    log = get_logger(__name__, rank_zero_only=True)

    @wraps(func)
    def _wrapper(*args, **kwargs) -> None:
        """Wrap function to check for positional arguments.

        Args:
        ----
            args: The positional arguments.
            kwargs: The keyword arguments

        """
        if len(args) != 1 if inspect.ismethod(func) else 0:
            log.warning(
                "WARNING: using %s with positional arguments is"
                " deprecated and will be disallowed in a future version of"
                " lmms-eval!",
                func.__name__,
            )
        return func(*args, **kwargs)

    return _wrapper


def rank_zero_only(func: Callable, default: Callable | None = None) -> Callable | None:
    """Call function only if the rank is zero.

    Args:
    ----
        func (Callable): Function to be wrapped
        default (Callable | None): Default value to return if the rank is not zero. Defaults to
            None.

    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs) -> Callable | None:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return func(*args, **kwargs)
        return default

    return wrapped_fn
