import logging
import os
from collections.abc import Callable
from functools import wraps

from src.utils._decorators import rank_zero_only as rank_zero_only_fn

__all__ = ["get_logger"]


def get_logger(name: str = __name__, rank_zero_only: bool = False) -> logging.Logger:
    """Initialize multi-GPU-friendly python command line logger.

    Args:
    ----
        name (str): The name of the logger, defaults to ``__name__``.
        rank_zero_only (bool): If True, the logger will only log on the process with rank 0.
            Defaults to False.

    """

    def rank_prefixed_log(func: "Callable") -> "Callable":
        """Add a prefix to a log message indicating its local rank.

        If `rank` is provided in the wrapped functions kwargs, then the log will only occur on
        that rank/process.

        Args:
        ----
            func (Callable): The function to wrap.

        """

        @wraps(func)
        def inner(
            *inner_args, rank_to_log: int | None = None, **inner_kwargs
        ) -> "Callable | None":
            rank = getattr(rank_zero_only_fn, "rank", None)
            if rank is None:
                rank_zero_only_fn.rank = os.getenv("LOCAL_RANK", 0)

            # Add the rank to the extra kwargs
            extra = inner_kwargs.pop("extra", {})
            extra.update({"rank": rank})

            if rank_zero_only:
                if rank == 0:
                    return func(*inner_args, extra=extra, **inner_kwargs)
            elif rank_to_log is None or rank == rank_to_log:
                return func(*inner_args, extra=extra, **inner_kwargs)
            else:
                return None

        return inner

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank_prefixed_log decorator
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_prefixed_log(getattr(logger, level)))

    # Set the logging format (with colors if available)
    try:
        import colorlog

        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "[%(cyan)s%(asctime)s%(reset)s][%(blue)srank:%(rank)s%(reset)s]"
            "[%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s"
        )
    except ImportError:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s][rank:%(rank)s][%(name)s][%(levelname)s] - %(message)s"
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Avoid propagating logs to the root logger
    logger.propagate = False

    return logger
