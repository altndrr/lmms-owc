from collections.abc import Callable

from src.data.filters._base import Filter, FilterEnsemble
from src.schema import FilterInfo

__all__ = [
    "FILTERS",
    "get_filter",
    "get_filter_builder",
    "get_filter_info",
    "get_filters_ensemble",
    "get_filters_info",
    "register_filter",
]

FILTERS: dict[str, FilterInfo] = {}


def get_filter(filter_id: str, **filter_kwargs) -> Filter:
    """Get a filter.

    Args:
    ----
        filter_id (str): The name of the filter.
        filter_kwargs (dict): Keyword arguments to pass to the filter builder.

    """
    return FILTERS[filter_id].builder_fn(**filter_kwargs)


def get_filter_builder(filter_id: str) -> Callable | None:
    """Get a filter builder.

    Args:
    ----
        filter_id (str): The name of the filter.

    """
    return FILTERS[filter_id].builder_fn


def get_filter_info(filter_id: str) -> FilterInfo:
    """Get the filter info.

    Args:
    ----
        filter_id (str): The name of the filter.

    """
    return FILTERS[filter_id]


def get_filters_ensemble(name: str, components: list[tuple[str, dict | None]]) -> FilterEnsemble:
    """Get an ensemble of filters.

    Args:
    ----
        name (str): The name of the filter ensemble.
        components (list[tuple[str, dict | None]]): The components of the filter.

    """
    filters = []
    for key, kwargs in components:
        filter_cls = key if isinstance(key, Callable) else get_filter_builder(key)
        filter = filter_cls(**kwargs) if kwargs else filter_cls()
        filters.append(filter)

    return FilterEnsemble(name=name, filters=filters)


def get_filters_info() -> list[FilterInfo]:
    """Get all filters info."""
    return list(FILTERS.values())


def register_filter(name: str | None = None) -> Callable:
    """Register a filter.

    Args:
    ----
        name (str, optional): The name of the filter.

    """

    def decorator(filter: Callable) -> Callable:
        FILTERS[name or filter.__name__.lower()] = FilterInfo(
            name=name or filter.__name__.lower(), builder_fn=filter
        )
        return filter

    return decorator
