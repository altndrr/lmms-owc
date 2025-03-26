from collections.abc import Callable

from src.data.samplers._base import Sampler
from src.schema import SamplerInfo

__all__ = [
    "SAMPLERS",
    "get_sampler",
    "get_sampler_builder",
    "get_sampler_info",
    "get_samplers_info",
    "register_sampler",
]

SAMPLERS: dict[str, SamplerInfo] = {}


def get_sampler(sampler_id: str, **sampler_kwargs) -> Sampler:
    """Get a sampler.

    Args:
    ----
        sampler_id (str): The name of the sampler.
        sampler_kwargs (dict): Keyword arguments to pass to the sampler builder.

    """
    return SAMPLERS[sampler_id].builder_fn(**sampler_kwargs)


def get_sampler_builder(sampler_id: str) -> Callable | None:
    """Get a sampler builder.

    Args:
    ----
        sampler_id (str): The name of the sampler.

    """
    return SAMPLERS[sampler_id].builder_fn


def get_sampler_info(sampler_id: str) -> SamplerInfo:
    """Get the sampler info.

    Args:
    ----
        sampler_id (str): The name of the sampler.

    """
    return SAMPLERS[sampler_id]


def get_samplers_info() -> list[SamplerInfo]:
    """Get all samplers info."""
    return list(SAMPLERS.values())


def register_sampler(name: str | None = None) -> Callable:
    """Register a sampler.

    Args:
    ----
        name (str, optional): The name of the sampler.

    """

    def decorator(sampler: Callable) -> Callable:
        SAMPLERS[name or sampler.__name__.lower()] = SamplerInfo(
            name=name or sampler.__name__.lower(), builder_fn=sampler
        )
        return sampler

    return decorator
