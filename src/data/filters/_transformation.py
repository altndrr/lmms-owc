from collections.abc import Iterable, Iterator

from src.data.filters._api import Filter, register_filter

__all__ = ["TRANSFORMATION_FILTERS", "LowercaseFilter", "MapFilter", "UppercaseFilter"]

TRANSFORMATION_FILTERS = ["lowercase", "map", "uppercase"]


@register_filter("lowercase")
class LowercaseFilter(Filter):
    """Filter used to convert the responses to lowercase."""

    def __init__(self) -> None:
        pass

    def apply(self, responses: list, docs: list | None = None) -> list:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """

        def filter_set(inst: list) -> list:
            """Filter a set of responses.

            Args:
            ----
                inst (list): The task instance.

            """
            return [resp.lower() for resp in inst]

        return [filter_set(resp) for resp in responses]


@register_filter("map")
class MapFilter(Filter):
    """Filter used to map responses to a given dictionary.

    The filter maps the responses to a given dictionary. If a response is not found in the
    dictionary, it returns a default value.

    Args:
    ----
        mapping_dict (dict): A dictionary containing the key-value mappings.
        default_value (Any): The value to be returned when a key is not found in the mapping_dict.

    """

    def __init__(
        self, mapping_dict: dict | None = None, default_value: int | float | str | None = None
    ) -> None:
        if mapping_dict is None:
            mapping_dict = {}

        self.mapping_dict = mapping_dict
        self.default_value = default_value

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """

        def filter_set(inst: list) -> list:
            """Filter a set of responses.

            Args:
            ----
                inst (list): The task instance.

            """
            return [self.mapping_dict.get(resp, self.default_value) for resp in inst]

        return [filter_set(resp) for resp in responses]


@register_filter("uppercase")
class UppercaseFilter(Filter):
    """Filter used to convert the responses to uppercase."""

    def __init__(self) -> None:
        pass

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """

        def filter_set(inst: list) -> list:
            """Filter a set of responses.

            Args:
            ----
                inst (list): The task instance.

            """
            return [resp.upper() for resp in inst]

        return [filter_set(resp) for resp in responses]
