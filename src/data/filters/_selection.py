from collections import Counter
from collections.abc import Iterable, Iterator

from src.data.filters._api import Filter, register_filter

__all__ = ["SELECTION_FILTERS", "MajorityVoteFilter", "TakeFirstFilter", "TakeKFilter"]

SELECTION_FILTERS = ["majority_vote", "take_first", "take_first_k"]


@register_filter("majority_vote")
class MajorityVoteFilter(Filter):
    """Filter used to select the most frequent response from a list of model responses."""

    def __init__(self) -> None:
        pass

    def apply(self, responses: list, docs: list | None = None) -> Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """

        def select_majority(response: list) -> str:
            counts = Counter(response)
            vote = counts.most_common(1)[0][0]
            return vote

        return map(lambda r: [select_majority(r)], responses)


@register_filter("take_first")
class TakeFirstFilter(Filter):
    """Filter used to select the first response from a list of model responses."""

    def __init__(self) -> None:
        pass

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """
        return map(lambda r: r[0], responses)


@register_filter("take_first_k")
class TakeKFilter(Filter):
    """Filter used to select the first k responses from a list of model responses."""

    def __init__(self, *args, **kwargs) -> None:
        self.k = kwargs.pop("k")

        super().__init__(*args, **kwargs)

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """
        if len(responses[0]) < self.k:
            raise ValueError(
                f"Need at least {self.k} responses per doc to take first {self.k}, but got"
                f" {len(responses[0])} only! Please increase TaskConfig.repeats ."
            )

        return map(lambda r: r[: self.k], responses)
