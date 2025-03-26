import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from inspect import getsource
from typing import Any

from src import utils

__all__ = ["AggregationConfig", "GroupConfig", "TaskConfig"]

log = utils.get_logger(__name__, rank_zero_only=True)


@dataclass
class AggregationConfig(dict):
    """Dataclass describing an aggregation metric configuration."""

    metric: str | None = None
    aggregation: str | None = "mean"
    weight_by_size: bool | None = False
    filter_list: str | list | None = "none"

    def __post_init__(self) -> None:
        """Post-init checks."""
        if self.aggregation != "mean" and not callable(self.aggregation):
            raise ValueError(
                "Currently, 'mean' is the only pre-defined aggregation across groups' subtasks."
                f" Got '{self.aggregation}'."
            )

        if isinstance(self.filter_list, str):
            self.filter_list = [self.filter_list]


@dataclass
class GroupConfig(dict):
    """Dataclass describing a group configuration."""

    group: str | None = None
    group_alias: str | None = None
    task: str | list | None = None
    aggregate_metric_list: list | AggregationConfig | dict | None = None
    metadata: dict | None = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )

    def __getitem__(self, item: str) -> Any:  # noqa: ANN401
        """Get an item from the configuration.

        Args:
        ----
            item (str): The item to get.

        """
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:  # noqa: ANN401
        """Set an item in the configuration.

        Args:
        ----
            item (str): The item to set.
            value (Any): The value to set.

        """
        return setattr(self, item, value)

    def __post_init__(self) -> None:
        """Post-init checks."""
        if self.aggregate_metric_list is not None:
            if isinstance(self.aggregate_metric_list, dict):
                self.aggregate_metric_list = list(self.aggregate_metric_list)

            self.aggregate_metric_list = [
                AggregationConfig(**item) if isinstance(item, dict) else item
                for item in self.aggregate_metric_list
            ]

    def to_dict(self, keep_callable: bool = False) -> dict:
        """Dump the current config as a dictionary object, as a printable format.

        Used for dumping results alongside full task configuration. Empty fields will not be
        printed.

        Args:
        ----
            keep_callable (bool, optional): Whether to keep the callable functions in the config.

        """
        cfg_dict = asdict(self)
        for k, v in list(cfg_dict.items()):
            if callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Callable | str, keep_callable: bool = False
    ) -> Callable | str:
        """Serialize a given function or string.

        Args:
        ----
            value (Callable | str): The function or string to serialize.
            keep_callable (bool, optional): Whether to keep the callable functions in the config.
                If True, the original callable is returned. Otherwise, attempts to return the
                source code of the callable using 'getsource'. Defaults to False.

        """
        if keep_callable:
            return value

        try:
            return getsource(value)
        except (TypeError, OSError):
            return str(value)


@dataclass
class TaskConfig(dict):
    """Dataclass describing a task configuration."""

    task: str | None = None
    task_alias: str | None = None
    tag: str | None = None
    group: str | None = None
    group_alias: str | list | None = None

    # Dataset options
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_kwargs: dict | None = None
    training_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    fewshot_split: str | None = None
    full_docs: bool = False

    # Formatting and prompting options
    process_results_use_image: bool = False
    process_docs: Callable | None = None
    doc_to_visual: Callable | str | None = None
    doc_to_text: Callable | str | None = None
    doc_to_target: Callable | str | None = None
    doc_to_choice: Callable | str | dict | list | None = None
    process_results: Callable | str | None = None
    use_prompt: str | None = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict | None = None

    # runtime options
    num_fewshot: int | None = None

    # Scoring options
    metric_list: list | None = None
    output_type: str = "generate_until"
    generation_kwargs: dict | None = None
    repeats: int = 1
    filter_list: str | list | None = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str | None = None

    # By default, not used in the code. allows for users to pass arbitrary info to tasks
    metadata: str | list | None = None

    model_specific_kwargs: dict | None = None
    model_specific_generation_kwargs: dict | None = None
    model_specific_target_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.dataset_path and os.path.exists(os.path.dirname(self.dataset_path)):
            pass

        if self.group is not None:
            log.warning(
                "A task YAML file was found to contain a `group` key. Groups which provide"
                " aggregate scores over several subtasks now require a separate config file--if"
                " not aggregating, you may want to use the `tag` config option instead within"
                " your config. Setting `group` within a TaskConfig will be deprecated in v0.4.4."
                " Please see https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md for more information."  # noqa: E501
            )

            if self.tag is None:
                self.tag = self.group
            else:
                raise ValueError(
                    "Got both a `group` and `tag` entry within a TaskConfig. Please use one or the"
                    " other--`group` values will be deprecated in v0.4.4."
                )

        if self.generation_kwargs is not None:
            if "generate_until" not in self.output_type:
                raise ValueError(
                    f"Task {self.task} has no `generation_kwargs` for `output_type:"
                    " generate_until`"
                )

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(
                    self.generation_kwargs["temperature"]
                )

            if "until" not in self.generation_kwargs:
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if "generate_until" in self.output_type:
                # ensure that we greedily generate in absence of explicit arguments otherwise
                self.generation_kwargs = {
                    "until": None if self.fewshot_delimiter is None else [self.fewshot_delimiter],
                    "do_sample": False,
                }

    def __getitem__(self, item: str) -> Any:  # noqa: ANN401
        """Get an item from the configuration.

        Args:
        ----
            item (str): The item to get.

        """
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:  # noqa: ANN401
        """Set an item in the configuration.

        Args:
        ----
            item (str): The item to set.
            value (Any): The value to set.

        """
        return setattr(self, item, value)

    def to_dict(self) -> dict:
        """Dump the current config as a dictionary object, as a printable format.

        Used for dumping results alongside full task configuration. Empty fields will not be
        printed.

        """
        cfg_dict = asdict(self)
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                cfg_dict[k] = str(v)
        return cfg_dict
