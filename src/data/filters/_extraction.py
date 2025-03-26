import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from typing import Any

from src.data.filters._api import Filter, register_filter

__all__ = ["EXTRACTION_FILTERS", "MultiChoiceRegexFilter", "RegexFilter", "WhitespaceFilter"]

EXTRACTION_FILTERS = ["regex", "multi_choice_regex", "remove_whitespace"]

unicode_punctuation_map = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


@register_filter("regex")
class RegexFilter(Filter):
    """Filter used to extract a regex pattern from a model response.

    Args:
    ----
        regex_pattern (str): The regex pattern to use.
        group_select (int): Selects the (group_select)th match from the findall result.
        fallback (str): The fallback value to use if no matches are found.

    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
    ) -> None:
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

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
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered

        filtered_responses = list(map(lambda x: filter_set(x), responses))
        return filtered_responses


@register_filter("multi_choice_regex")
class MultiChoiceRegexFilter(RegexFilter):
    """Filter used to extract a model's answer on multiple choice questions.

    It assumes each document has a "choices" field containing the list of answer choices and that
    the answer label symbols are of the form (A), (B), (C), ... or A, B, C.

    Args:
    ----
        regex_pattern (str): The regex pattern to use.
        group_select (int): Selects the (group_select)th match from the findall result.
        fallback (str): The fallback value to use if no matches are found.
        ignore_case (bool): Ignore the case during step 1 matching.
        ignore_punctuation (bool): Remove the punctuation during step 1 matching.
        regexes_to_ignore (list): Remove these regexes during step 1 matching.

    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select: int = 0,
        fallback: str = "[invalid]",
        ignore_case: bool = False,
        ignore_punctuation: bool = False,
        regexes_to_ignore: list | None = None,
    ) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def find_match(self, regex: re.Pattern, resp: str, convert_dict: dict | None = None) -> Any:  # noqa: ANN401
        """Find a match in the response.

        Args:
        ----
            regex (re.Pattern): The regex pattern.
            resp (str): The response.
            convert_dict (dict): The conversion dictionary.

        """
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match

    def filter_ignores(self, st: str) -> str:
        """Filter the ignores.

        Args:
        ----
            st (str): The string.

        """
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, "", st)

        if self.ignore_case:
            st = st.lower()

        if self.ignore_punctuation:
            st = st.translate(unicode_punctuation_map)  # https://stackoverflow.com/a/266162
        return st

    def apply(self, responses: list, docs: list | None = None) -> Iterable | Iterator:
        """Apply the filter to the responses of the model.

        Args:
        ----
            responses (list): List of responses from the model.
            docs (list, optional): List of documents. Defaults to None.

        """
        # Here, we assume we have a list, in which each element is a list of model responses for
        # some particular input/target pair. So we process each of these (same input/target
        # response sets) independently (and keep them a list).

        filtered_responses = []
        for r, doc in zip(responses, docs, strict=True):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for c in choices:
                m = self.filter_ignores(c.strip())
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = f"({next_alpha})"

                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f"({next_alpha})"

                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile("|".join(fallback_regexes))
            without_paren_fallback_regex = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(f":[\\s]*({without_paren_fallback_regex})")

            filtered = []
            convert_dict = {}
            for resp in r:
                match = self.find_match(self.regex, resp, convert_dict)
                if not match:
                    match = self.find_match(
                        fallback_regex, self.filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        match = self.find_match(
                            without_paren_fallback_regex, resp, without_paren_to_target
                        )
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_responses.append(filtered)

        return filtered_responses


@register_filter("remove_whitespace")
class WhitespaceFilter(Filter):
    """Filter used to remove leading whitespace from model responses."""

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
            filtered_resp = []
            for resp in inst:
                if resp.startswith(" "):
                    resp = resp[1:]

                filtered_resp.append(resp)

            return filtered_resp

        filtered_responses = [filter_set(resp) for resp in responses]
        return filtered_responses
