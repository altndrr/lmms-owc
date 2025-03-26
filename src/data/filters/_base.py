from collections.abc import Iterable, Iterator

__all__ = ["Filter", "FilterEnsemble"]


class Filter:
    """Class describing a filter.

    It operates on a per-task level. It takes all model outputs (`instance.resps` for all
    `task.instances`) across all instances of a task, and perform operations. In a single run, one
    can configure any number of separate filters or lists of filters.

    Args:
    ----
        args: Any positional arguments to be passed to the filter.
        kwargs: Any keyword arguments to be passed to the filter.

    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """
        return responses


class FilterEnsemble:
    """Class to create a filtering pipeline.

    It applies multiple filters in sequence. Its intended usage is to stack multiple post-
    processing steps in order.
    """

    def __init__(self, name: str, filters: list[Filter]) -> None:
        self.name = name
        self.filters = filters

    def apply(self, instances: list, docs: list | None = None) -> None:
        """Apply the filter ensemble to the responses of the model.

        Args:
        ----
            instances (list): List of instances.
            docs (list, optional): List of documents. Defaults to None.

        """
        responses = [inst.resps for inst in instances]
        for f in self.filters:
            responses = f.apply(responses, docs)

        for inst, response in zip(instances, responses, strict=True):
            inst.filtered_resps[self.name] = response
