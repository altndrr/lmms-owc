import ast
import collections
import copy
import inspect
import os
import random
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from typing import Any

import datasets
import numpy as np
from datasets import DownloadConfig, Image, Sequence
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed

from src import utils
from src.data.filters import get_filters_ensemble
from src.data.metrics import (
    DEFAULT_METRICS_PER_OUTPUT_TYPE,
    get_aggregation_builder,
    get_metric_builder,
    get_metric_info,
)
from src.data.samplers import get_sampler_builder
from src.data.tasks._base import Task, TaskInstance
from src.data.tasks._config import GroupConfig, TaskConfig

__all__ = ["GroupConfig", "TaskConfig", "TaskManager"]

log = utils.get_logger(__name__, rank_zero_only=True)

ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "generate_until",
    "generate_until_multi_round",
]
GROUP_ONLY_KEYS = list(GroupConfig().to_dict().keys())


def _class_has_config_in_constructor(cls: type) -> bool:
    """Check if a class has a config in the constructor.

    Args:
    ----
        cls (type): The class to check.

    """
    constructor = getattr(cls, "__init__", None)
    return "config" in inspect.signature(constructor).parameters if constructor else False


def _config_is_task(config: dict) -> bool:
    """Check if a config is a task.

    Args:
    ----
        config (dict): The config to check.

    """
    return bool("task" in config and isinstance(config["task"], str))


def _config_is_group(config: dict) -> bool:
    """Check if a config is a group.

    Args:
    ----
        config (dict): The config to check.

    """
    return bool("task" in config and isinstance(config["task"], list))


def _config_is_python_task(config: dict) -> bool:
    """Check if a config is a python task.

    Args:
    ----
        config (dict): The config to check.

    """
    return "class" in config


class ConfigurableGroup:
    """Class for a configurable group."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = GroupConfig(**config)

    @property
    def group(self) -> str:
        """Get the group."""
        return self._config.group

    @property
    def group_alias(self) -> str:
        """Get the group alias."""
        return self._config.group_alias

    @property
    def config(self) -> dict:
        """Get the configuration."""
        return self._config.to_dict()

    @property
    def group_name(self) -> str:
        """Get the group name."""
        return self._config.group

    def __repr__(self) -> str:
        """Get the representation of the configurable group."""
        return f"ConfigurableGroup(group={self.group}," f"group_alias={self.group_alias})"


class ConfigurableTask(Task):
    """A configurable evaluation task.

    A task represents an entire benchmark including its dataset, problems, answers, and evaluation
    methods. See BoolQ for a simple example implementation.

    A `doc` can be any python object which represents one instance of evaluation. This is usually a
    dictionary, e.g., {"question": ..., "answer": ...} or {"question": ..., question, answer}.

    Args:
    ----
        data_dir (str, optional): The path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when the dataset is
            not publicly accessible). Defaults to None.
        cache_dir (str, optional): The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
            `~/.cache/huggingface/datasets`. Defaults to None.
        download_mode (datasets.DownloadMode, optional): How to treat pre-existing `Task` downloads
            and data. Defaults to None.
        config (dict, optional): A dictionary specifying the configuration of the task. Defaults to
            None.
        model_name (str, optional): The name of the model to use for the task. Defaults to None.

    """

    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    dataset: datasets.Dataset

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: datasets.DownloadMode | None = None,
        config: dict | None = None,
        model_name: str | None = None,
    ) -> None:
        # ! no super() call

        # Get pre-configured attributes
        self._config = self.CONFIG

        if self.config is None:
            self._config = TaskConfig(**config)
        elif config is not None:
            self._config.__dict__.update(config)

        if self.config is None:
            raise ValueError(
                "Must pass a config to ConfigurableTask, either in cls.CONFIG or `config` kwarg"
            )

        if isinstance(self.config.metadata, dict) and "version" in self.config.metadata:
            self.VERSION = self.config.metadata["version"]

        self.model_name = model_name
        self._prepare_model_specific_config()

        if self.config.output_type is not None:
            if self.config.output_type not in ALL_OUTPUT_TYPES:
                raise ValueError(
                    f"Got invalid output_type '{self.config.output_type}', must be"
                    f" in '{','.join(ALL_OUTPUT_TYPES)}'"
                )
            self.OUTPUT_TYPE = self.config.output_type

        if self.config.dataset_path is not None:
            self.DATASET_PATH = self.config.dataset_path

        if self.config.dataset_name is not None:
            self.DATASET_NAME = self.config.dataset_name

        self._prepare_metric_and_aggregation()

        self.download(self.config.dataset_kwargs)
        self._training_docs = None
        self._fewshot_docs = None

        if self.config.filter_list is not None:
            self._filters = []
            for filter_config in self.config.filter_list:
                for filter_pipeline in filter_config:
                    filter_name = filter_config["name"]
                    filter_functions = filter_config["filter"]
                    components = []
                    for function in filter_functions:
                        kwargs = {key: function[key] for key in function if key != "function"}
                        components.append((function["function"], kwargs))
                    filter_pipeline = get_filters_ensemble(filter_name, components)
                self._filters.append(filter_pipeline)
        else:
            self._filters = [get_filters_ensemble("none", [("take_first", None)])]

        if self.config.fewshot_config is not None:
            sampler_key = (
                self.config.fewshot_config.get("sampler", "default")
                if self.config.fewshot_config
                else "default"
            )
            sampler_cls = get_sampler_builder(sampler_key)
            self.sampler = sampler_cls(list(self.fewshot_docs()), self, rnd=random.Random(1234))  # noqa: S311

        if self.has_test_docs():
            self.task_docs = self.test_docs()
        elif self.has_validation_docs():
            self.task_docs = self.validation_docs()
        else:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have"
                " valid or test docs!"
            )

        # Test One Doc
        self.features = list(self.task_docs.features.keys())
        self.multiple_input = 0
        self.multiple_target = 0
        test_doc = self.task_docs[0]
        test_text = self.doc_to_text(test_doc)
        test_target = self.doc_to_target(test_doc)

        if self.config.doc_to_choice is not None:
            test_choice = self.doc_to_choice(test_doc)
            if not isinstance(test_choice, list):
                log.error("doc_to_choice must return list")
            else:
                num_choice = len(test_choice)

            if isinstance(test_text, int):
                self.multiple_input = num_choice
        else:
            test_choice = None

        if isinstance(test_target, list):
            self.multiple_target = len(test_target)
        else:
            if isinstance(test_target, int) and test_choice is not None:
                test_target = test_choice[test_target]
            else:
                test_target = str(test_target)

        check_choices = test_choice if test_choice is not None else [test_target]
        if self.config.doc_to_choice is not None:
            for choice in check_choices:
                choice_has_whitespace = bool(choice[0].isspace())
                delimiter_has_whitespace = (
                    self.config.target_delimiter.rstrip() != self.config.target_delimiter
                )

                if delimiter_has_whitespace and choice_has_whitespace:
                    log.warning(
                        'Both target_delimiter and target choice: "%s" have whitespace', choice
                    )
                elif (not delimiter_has_whitespace) and (not choice_has_whitespace):
                    log.warning(
                        'Both target_delimiter "%s" and target choice: "%s" do not have'
                        " whitespace, ignore if the language you are evaluating on does not"
                        " require/use whitespace",
                        self.config.target_delimiter,
                        choice,
                    )

    def _prepare_model_specific_config(self) -> None:
        """Prepare model-specific configs."""
        self.model_specific_kwargs = self.config.model_specific_kwargs
        if self.model_specific_kwargs is not None:
            if self.model_name in self.model_specific_kwargs:
                self.model_specific_kwargs = self.model_specific_kwargs[self.model_name]
            elif "default" in self.model_specific_kwargs:
                self.model_specific_kwargs.update(self.model_specific_kwargs.get("default", {}))
            elif "dataset" in self.model_specific_kwargs:
                self.model_specific_kwargs.update(self.model_specific_kwargs.get("dataset", {}))

        self.model_specific_target_kwargs = self.config.model_specific_target_kwargs
        if self.model_specific_target_kwargs is not None:
            if self.model_name in self.model_specific_target_kwargs:
                self.model_specific_target_kwargs = self.model_specific_target_kwargs[
                    self.model_name
                ]
            else:
                self.model_specific_target_kwargs = self.model_specific_target_kwargs.get(
                    "default", None
                )
        self.model_specific_generation_kwargs = self.config.model_specific_generation_kwargs
        if self.model_specific_generation_kwargs is not None:
            if self.model_name in self.model_specific_generation_kwargs:
                self.model_specific_generation_kwargs = self.model_specific_generation_kwargs[
                    self.model_name
                ]
            else:
                self.model_specific_generation_kwargs = self.model_specific_generation_kwargs.get(
                    "default", {}
                )

            self.config.generation_kwargs.update(self.model_specific_generation_kwargs)

    def _prepare_metric_and_aggregation(self) -> None:
        """Prepare metric and aggregation configs."""
        self._metric_fn_list = {}
        self._metric_fn_kwargs = {}
        self._aggregation_list = {}
        self._higher_is_better = {}

        if self.config.metric_list is None:
            # TODO handle this in TaskConfig.__post_init__ ?
            _metric_list = DEFAULT_METRICS_PER_OUTPUT_TYPE[self.config.output_type]

            for metric_name in _metric_list:
                metric_info = get_metric_info(metric_name)
                self._metric_fn_list[metric_name] = metric_info.builder_fn
                self._metric_fn_kwargs[metric_name] = {}
                self._aggregation_list[metric_name] = metric_info.group_fn
                self._higher_is_better[metric_name] = metric_info.higher_is_better
        else:
            for metric_config in self.config.metric_list:
                if "metric" not in metric_config:
                    raise KeyError("Missing required 'metric' key in metric configuration")
                metric_name = metric_config["metric"]
                kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key not in ["metric", "aggregation", "higher_is_better"]
                }

                if self.config.process_results is not None:
                    self._metric_fn_list[metric_name] = None
                    self._metric_fn_kwargs[metric_name] = {}
                elif callable(metric_name):
                    metric_fn = metric_name.__call__
                    metric_name = metric_name.__name__
                    self._metric_fn_list[metric_name] = metric_fn
                    self._metric_fn_kwargs[metric_name] = kwargs
                else:
                    self._metric_fn_list[metric_name] = get_metric_builder(metric_name)
                    self._metric_fn_kwargs[metric_name] = kwargs

                if "aggregation" in metric_config:
                    agg_name = metric_config["aggregation"]
                    if isinstance(agg_name, str):
                        self._aggregation_list[metric_name] = get_aggregation_builder(agg_name)
                    elif callable(agg_name):
                        self._aggregation_list[metric_name] = metric_config["aggregation"]
                else:
                    metric_agg = get_metric_info(metric_name).group_fn
                    log.warning(
                        "In task %s, metric %s is defined, but aggregation is not. "
                        "Using default aggregation=%s",
                        self._config.task,
                        metric_name,
                        metric_agg.__name__,
                    )
                    self._aggregation_list[metric_name] = metric_agg

                if "higher_is_better" in metric_config:
                    self._higher_is_better[metric_name] = metric_config["higher_is_better"]
                else:
                    higher_is_better = get_metric_info(metric_name).higher_is_better
                    log.warning(
                        "In task %s, metric %s is defined, but higher_is_better is not. "
                        "using default higher_is_better=%s",
                        self._config.task,
                        metric_name,
                        higher_is_better,
                    )
                    self._higher_is_better[metric_name] = higher_is_better

    @retry(stop=(stop_after_attempt(5) | stop_after_delay(60)), wait=wait_fixed(2))
    def download(self, dataset_kwargs: dict | None = None) -> None:
        """Download and prepare the dataset.

        Args:
        ----
            dataset_kwargs (dict, optional): Parameters to be passed to the dataset-specific
                `download` function.

        """
        if dataset_kwargs is None:
            dataset_kwargs = {}

        download_config = DownloadConfig()
        download_config.max_retries = dataset_kwargs.get("max_retries", 10)
        download_config.num_proc = dataset_kwargs.get("num_proc", 8)
        download_config.local_files_only = dataset_kwargs.get("local_files_only", False)

        if "force_download" in dataset_kwargs:
            dataset_kwargs.pop("force_download")

        if "force_unzip" in dataset_kwargs:
            dataset_kwargs.pop("force_unzip")

        if "local_files_only" in dataset_kwargs:
            dataset_kwargs.pop("local_files_only")

        if "create_link" in dataset_kwargs:
            dataset_kwargs.pop("create_link")

        if dataset_kwargs.get("load_from_disk", False):
            dataset_kwargs.pop("load_from_disk")

            if isinstance(dataset_kwargs.get("custom_download", None), Callable):
                custom_download_fn = dataset_kwargs["custom_download"]
                custom_download_fn()

            self.dataset = datasets.load_from_disk(self.DATASET_PATH)
        else:
            self.dataset = datasets.load_dataset(
                path=self.DATASET_PATH,
                name=self.DATASET_NAME,
                download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                download_config=download_config,
                **dataset_kwargs if dataset_kwargs is not None else {},
            )

        if self.config.process_docs is not None:
            for split in self.dataset:
                if split in [
                    self.config.training_split,
                    self.config.validation_split,
                    self.config.test_split,
                    self.config.fewshot_split,
                ]:
                    self.dataset[split] = self.config.process_docs(self.dataset[split])

        # Copy dataset, remove image features
        self.dataset_no_image = self.dataset.copy()
        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features
            # If it is an Image instance or a Sequence of Image instance. Remove it
            for feature in features:
                if (
                    isinstance(features[feature], Image)
                    or isinstance(features[feature], Sequence)
                    and isinstance(features[feature].feature, Image)
                ):
                    remove_cols.append(feature)
            for remove_col in remove_cols:
                self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(
                    remove_col
                )

    def has_training_docs(self) -> bool:
        """Whether the task has a training set."""
        return self.config.training_split is not None

    def has_validation_docs(self) -> bool:
        """Whether the task has a validation set."""
        return self.config.validation_split is not None

    def has_test_docs(self) -> bool:
        """Whether the task has a test set."""
        return self.config.test_split is not None

    def training_docs(self) -> datasets.Dataset:
        """Get the training docs for the `doc_to_text` method."""
        if self.has_training_docs():
            return self.dataset[self.config.training_split]

    def validation_docs(self) -> datasets.Dataset:
        """Get the validation docs for the `doc_to_text` method."""
        if self.has_validation_docs():
            return self.dataset[self.config.validation_split]

    def test_docs(self) -> datasets.Dataset:
        """Get the test docs for the `doc_to_text` method."""
        if self.has_test_docs():
            return self.dataset[self.config.test_split]

    def fewshot_docs(self) -> datasets.Dataset:
        """Get the fewshot docs for the `doc_to_text` method."""
        if self.config.fewshot_split is not None:
            return self.dataset[self.config.fewshot_split]
        else:
            if (self.config.num_fewshot is not None) and (self.config.num_fewshot > 0):
                log.warning(
                    "Task '%s': num_fewshot > 0 but fewshot_split is None. Using pre-configured"
                    " rule.",
                    self.config.task,
                )
            return super().fewshot_docs()

    @utils.deprecated_positional
    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable | None = None,
    ) -> str | list:
        """Generate the fewshot context string.

        The context is made up of a prepended description (if provided), the `num_fewshot` number
        of examples, and an appended prompt example.

        Args:
        ----
            doc (dict): The document to create a context for.
            num_fewshot (int): The number of fewshot examples to provide in the returned context
                string.
            system_instruction (str, optional): The system instructions to prepend to the
                context. Defaults to None.
            apply_chat_template (bool, optional): Whether to apply the chat template to the
                context. Defaults to False.
            fewshot_as_multiturn (bool, optional): Whether to format the fewshot examples as
                multi-turn chat examples. Defaults to False.
            chat_template (Callable, optional): The chat template to apply to the context.
                Defaults to None.

        """
        # Get task description
        if description := self.config.description:
            description = utils.apply_jinja_template(self.config.description, doc)

        # Create system prompt based on the provided system instruction and description
        if system_instruction is not None and description:
            system_prompt = f"{system_instruction}{self.sampler.fewshot_delimiter}{description}"
        elif system_instruction is not None:
            system_prompt = system_instruction
        elif description:
            system_prompt = description
        else:
            system_prompt = ""

        if apply_chat_template:
            labeled_examples = []

            # Add system prompt if specified
            if system_prompt:
                labeled_examples.append({"role": "system", "content": system_prompt})

            # If few-shot - append examples after the system prompt
            if num_fewshot > 0:
                labeled_examples.extend(
                    self.sampler.get_chat_context(doc, num_fewshot, fewshot_as_multiturn)
                )

            example = self.doc_to_text(doc)
            if self.multiple_input:
                return chat_template(labeled_examples)
            if isinstance(example, str):
                self.append_target_question(labeled_examples, example, fewshot_as_multiturn)
            # For loglikelihood create a list of questions with appended choices
            elif isinstance(example, list):
                labeled_examples_list = []
                # Copy chat history for each example and append the answer
                for ex in example:
                    chat = deepcopy(labeled_examples)
                    self.append_target_question(chat, ex, fewshot_as_multiturn)
                    labeled_examples_list.append(chat_template(chat))
                return labeled_examples_list
            # If example is an integer, append the choice or convert to string
            elif isinstance(example, int):
                if self.config.doc_to_choice is not None:
                    choices = self.doc_to_choice(doc)
                    self.append_target_question(
                        labeled_examples, choices[example], fewshot_as_multiturn
                    )
                else:
                    self.append_target_question(
                        labeled_examples, str(example), fewshot_as_multiturn
                    )

            return chat_template(labeled_examples)
        else:
            labeled_examples = ""

            # Add system prompt if specified
            if system_prompt:
                labeled_examples += system_prompt

            # If few-shot - append examples after the system prompt
            if num_fewshot > 0:
                labeled_examples += self.sampler.get_context(doc, num_fewshot)

            example = self.doc_to_text(doc)
            if self.multiple_input:
                return labeled_examples
            if isinstance(example, str):
                return labeled_examples + example
            elif isinstance(example, list):
                return [labeled_examples + ex for ex in example]
            elif isinstance(example, int):
                if self.config.doc_to_choice is not None:
                    choices = self.doc_to_choice(doc)
                    return labeled_examples + choices[example]
                return labeled_examples + str(example)
            else:
                raise ValueError("Unknown example type.")

    def apply_filters(self) -> list | None:
        """Apply filters to the instances."""
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances, self.task_docs)
        else:
            log.warning("No filter defined, passing through instances")
            return self._instances

    def should_decontaminate(self) -> bool:
        """Whether to decontaminate the dataset."""
        return self.config.should_decontaminate

    def doc_to_decontamination_query(self, doc: dict) -> str | None:
        """Get the decontamination query for the `doc_to_text` method.

        Override this method with specific decontamination query.

        Args:
        ----
            doc (dict): A single document.

        """
        if self.config.should_decontaminate:
            if self.config.doc_to_decontamination_query is None:
                return self.doc_to_text(doc)
            else:
                doc_to_decontamination_query = self.config.doc_to_decontamination_query
                if doc_to_decontamination_query in self.features:
                    return doc[doc_to_decontamination_query]
                elif callable(doc_to_decontamination_query):
                    return doc_to_decontamination_query(doc)
                else:
                    return ast.literal_eval(
                        utils.apply_jinja_template(self.config.doc_to_decontamination_query, doc)
                    )

    def doc_to_text(self, doc: dict) -> str:
        """Get the text for the `doc_to_text` method.

        Args:
        ----
            doc (dict): A single document.

        """
        doc_to_text = self.config.doc_to_text

        if isinstance(doc_to_text, int):
            return doc_to_text
        elif isinstance(doc_to_text, str):
            if doc_to_text in self.features:
                return doc[doc_to_text]
            else:
                text_string = utils.apply_jinja_template(doc_to_text, doc)
                if text_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(text_string)
                else:
                    return text_string
        elif callable(doc_to_text):
            return (
                doc_to_text(doc, self.model_specific_kwargs)
                if self.model_specific_kwargs is not None
                else doc_to_text(doc)
            )

        # Used when applying a Promptsource template
        elif hasattr(doc_to_text, "apply"):
            applied_prompt = doc_to_text.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[0]
            else:
                log.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            raise TypeError(
                "doc_to_text must be a string, int, or callable, got %s", type(doc_to_text)
            )

    def doc_to_target(self, doc: dict) -> int | str | list:
        """Get the target for the `doc_to_text` method.

        Args:
        ----
            doc (dict): A single document.

        """
        doc_to_target = self.config.doc_to_target

        if isinstance(doc_to_target, int):
            return doc_to_target
        elif isinstance(doc_to_target, str):
            if doc_to_target in self.features:
                return doc[doc_to_target]
            else:
                target_string = utils.apply_jinja_template(doc_to_target, doc)
                if target_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(target_string)
                elif (
                    len(target_string) >= 2
                    and (target_string[0] == "[")
                    and (target_string[-1] == "]")
                ):
                    try:
                        return ast.literal_eval(target_string)
                    except (SyntaxError, ValueError):
                        return target_string
                else:
                    return target_string
        elif isinstance(doc_to_target, list):
            return doc_to_target
        elif callable(doc_to_target):
            return (
                doc_to_target(doc, self.model_specific_target_kwargs)
                if self.model_specific_target_kwargs is not None
                else doc_to_target(doc)
            )

        # Used when applying a Promptsource template
        elif hasattr(doc_to_target, "apply"):
            applied_prompt = doc_to_target.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[1]
            else:
                log.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            raise TypeError(
                "doc_to_target must be a string, list, or callable. Got %s", type(doc_to_target)
            )

    def doc_to_visual(self, doc: dict) -> int | str | list:
        """Get the target for the `doc_to_visual` method.

        Args:
        ----
            doc (dict): A single document.

        """
        if isinstance(self.config.doc_to_visual, str):
            if self.config.doc_to_visual not in self.features:
                raise ValueError(
                    f"doc_to_visual '{self.config.doc_to_visual}' not found in features:"
                    f" {self.features}"
                )

            # Single image. Still return a list for consistency.
            return [doc[self.config.doc_to_visual]]
        elif callable(self.config.doc_to_visual):
            return (
                self.config.doc_to_visual(doc, self.model_specific_kwargs)
                if self.model_specific_kwargs is not None
                and len(inspect.signature(self.config.doc_to_visual).parameters) == 2
                else self.config.doc_to_visual(doc)
            )
        else:
            return self.config.doc_to_visual

    def doc_to_choice(self, doc: dict) -> list[str]:
        """Get the target for the `doc_to_choice` method.

        Args:
        ----
            doc (dict): A single document.

        """
        if self.config.doc_to_choice is None:
            log.error("`doc_to_choice` was called but not set in config.")
        else:
            doc_to_choice = self.config.doc_to_choice

        if isinstance(doc_to_choice, str):
            if doc_to_choice in self.features:
                return doc[doc_to_choice]
            else:
                return ast.literal_eval(utils.apply_jinja_template(doc_to_choice, doc))
        elif isinstance(doc_to_choice, list):
            return doc_to_choice
        elif isinstance(doc_to_choice, dict):
            return list(doc_to_choice.values())
        elif callable(doc_to_choice):
            return (
                doc_to_choice(doc, self.model_specific_kwargs)
                if self.model_specific_kwargs is not None
                and len(inspect.signature(doc_to_choice).parameters) == 2
                else doc_to_choice(doc)
            )
        elif hasattr(doc_to_choice, "get_answer_choices_list"):
            return doc_to_choice.get_answer_choices_list(doc)
        else:
            raise TypeError(
                f"`doc_to_choice` must be a string, list, dict, or callable,"
                f" but got {type(doc_to_choice)}"
            )

    def construct_requests(
        self, doc_id: int, ctx: str, **kwargs
    ) -> list[TaskInstance] | TaskInstance:
        """Construct Requests and returns an iterable of Requests which will be sent to the LMM.

        Args:
        ----
            doc_id (int): The index of a document within `self.test_docs()` or
                `self.validation_docs()`, whichever is the main split used.
            ctx (str): The context string, generated by fewshot_context. This includes the natural
                language description, as well as the few shot examples, and the question part of
                the document for `doc`.
            kwargs (dict): Additional keyword arguments.

        """
        split = kwargs["metadata"].get("split")

        if self.OUTPUT_TYPE == "loglikelihood":
            arguments = (
                ctx,
                self.doc_to_target,
                self.doc_to_visual,
                doc_id,
                self.config.task,
                split,
            )
        elif self.OUTPUT_TYPE == "multiple_choice":
            doc = self.dataset[split][doc_id]
            choices = self.doc_to_choice(doc)
            target_delimiter = self.config.target_delimiter
            if self.multiple_input:
                # If there are multiple inputs, choices are placed in the ctx
                cont = self.doc_to_target(doc)
                arguments = [
                    (
                        ctx,
                        f"{target_delimiter}{cont}",
                        self.doc_to_visual,
                        doc_id,
                        self.config.task,
                        split,
                    )
                    for ctx in choices
                ]
            else:
                # Otherwise they are placed in the continuation
                arguments = [
                    (
                        ctx,
                        f"{target_delimiter}{cont}",
                        self.doc_to_visual,
                        doc_id,
                        self.config.task,
                        split,
                    )
                    for cont in choices
                ]
            request_list = [
                TaskInstance(
                    request_type="loglikelihood",
                    # doc=doc,
                    arguments=arg,
                    idx=i,
                    **kwargs,
                )
                for i, arg in enumerate(arguments)
            ]
            # TODO should raise a warning telling users this will at most ~2x runtime.
            if "acc_mutual_info" in self._metric_fn_list:
                # If we are calculating multiple choice accuracy using mutual information instead
                # of raw loglikelihood as metric, need unconditional lls.

                # Here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # In other words normalizing by subtracting the unconditional logprob of each
                # choice.
                request_list.extend(
                    [
                        TaskInstance(
                            request_type="loglikelihood",
                            # doc=doc,
                            arguments=("", f"{choice}"),
                            idx=i,
                            **kwargs,
                        )
                        for i, choice in enumerate(choices)
                    ]
                )
            return request_list

        elif self.OUTPUT_TYPE == "generate_until":
            arguments = (
                ctx,
                copy.deepcopy(self.config.generation_kwargs),
                self.doc_to_visual,
                doc_id,
                self.config.task,
                split,
            )
        elif self.OUTPUT_TYPE == "generate_until_multi_round":
            arguments = (
                ctx,
                copy.deepcopy(self.config.generation_kwargs),
                self.doc_to_visual,
                partial(
                    self.config.doc_to_text,
                    model_specific_kwargs=self.model_specific_kwargs,
                ),
                doc_id,
                self.config.task,
                split,
            )
        return TaskInstance(request_type=self.OUTPUT_TYPE, arguments=arguments, idx=0, **kwargs)

    # TODO add a full_docs interface here for some evaluations that needs to access the full
    # datasets during process_results function. we may have better ways to handle this.
    @retry(stop=(stop_after_attempt(5) | stop_after_delay(1200)), wait=wait_fixed(2))
    def process_results(
        self, doc: dict, results: dict, full_docs: dict | None = None, **kwargs
    ) -> dict:
        """Evaluate the prediction of an LMM on the input.

        It takes in the document, the results of the requests, and any other relevant information
        and returns a dictionary where keys are the names of sub-metrics and values are the
        values of the metric for that one document.

        Args:
        ----
            doc (dict): The document as returned from training_docs, validation_docs, or test_docs.
            results (dict): The results of the requests created in construct_requests.
            full_docs (dict): The full document, including the context, if available.
            kwargs (dict): A dictionary of additional data, such as the LMM generation.

        """
        if self.OUTPUT_TYPE == "generate_until":
            if isinstance(results, list) and isinstance(results[0], list):
                results = [res.strip() for res in results[0]]
            else:
                results = [res.strip() for res in results]

        kwargs = {}
        if full_docs is not None:
            kwargs["full_docs"] = full_docs
        if callable(self.config.process_results):
            return self.config.process_results(doc, results, **kwargs)

        result_dict = {}
        use_metric = list(self._metric_fn_list.keys())
        if self.OUTPUT_TYPE == "loglikelihood":
            ll, is_greedy = results
            return {
                **({"perplexity": ll} if "perplexity" in use_metric else {}),
                **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
            }
        elif self.OUTPUT_TYPE == "multiple_choice":
            lls, is_greedy = zip(*results, strict=True)

            # Retrieve choices in list[str] form, to compute choice lengths, etc.
            choices = self.doc_to_choice(doc)
            completion_len = np.array([float(len(i)) for i in choices])

            if 2 * len(choices) == len(lls) and "acc_mutual_info" in self._metric_fn_list:
                # Then do mutual info. This stores the "dry-run" / unconditional answer
                # loglikelihoods
                lls_unconditional = lls[1::2]
                if len(lls_unconditional) != len(choices):
                    raise ValueError(
                        "Length mismatch between unconditional loglikelihoods"
                        f" ({len(lls_unconditional)}) and choices ({len(choices)})"
                    )
                # And this stores our "regular" conditional loglikelihoods
                lls = lls[::2]

            # ! Here may be different from original lm-eval since we return the actual loss in
            # many model loglikelihood. We just use the argmin here
            pred = np.argmin(lls)
            pred_norm = np.argmin(lls / completion_len)

            gold = self.doc_to_text(doc) if self.multiple_input else self.doc_to_target(doc)
            gold_index_error = False
            if isinstance(gold, list):
                gold = [i if i < len(choices) else -100 for i in gold]
                if -100 in gold:
                    gold_index_error = True
            else:
                if isinstance(gold, int):
                    gold = gold if gold < len(choices) else -100
                elif isinstance(gold, str):
                    gold = choices.index(gold) if gold in choices else -100

                if gold == -100:
                    gold_index_error = True

            if gold_index_error:
                log.warning(
                    "Label index was not in within range of available choices,"
                    "Sample:\n\n%s\n\n",
                    doc,
                )

            if self.multiple_target:
                acc = 1.0 if pred in gold else 0.0
                acc_norm = 1.0 if pred_norm in gold else 0.0
                exact_match = int(any([is_greedy[i] if i != -100 else 0 for i in gold]))
            else:
                acc = 1.0 if pred == gold else 0.0
                acc_norm = 1.0 if pred_norm == gold else 0.0
                # TODO this gets score of 0 on arc_challenge for pythia-70m. need to test that
                # this works properly
                exact_match = int(is_greedy[gold]) if gold != -100 else 0

            result_dict = {
                **({"acc": acc} if "acc" in use_metric else {}),
                **({"f1": (gold, pred)} if "f1" in use_metric else {}),
                **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
                **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
                **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
            }

            if "acc_mutual_info" in use_metric:
                lls_mutual_info = [
                    ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
                ]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif "generate_until" in self.OUTPUT_TYPE:
            gold = self.doc_to_target(doc)

            # For multi-turn, the results are a list, and we take the last one.
            if self.OUTPUT_TYPE == "generate_until_multi_round":
                result = [res[-1].strip() for res in results]
            else:
                result = [res.strip() for res in results]

            if self.config.doc_to_choice is not None:
                # If you set doc_to_choice, it assumes that doc_to_target returns a number.
                choices = self.doc_to_choice(doc)
                gold = choices[gold]
            # We expect multiple_targets to be a list.
            elif self.multiple_target:
                gold = list(gold)
            # If we have a single target but multiple result, we are probably in multi-round
            # and we select the last c.
            elif not self.multiple_target and isinstance(result, tuple):
                result = result[-1]

            for metric in self._metric_fn_list:
                if self.multiple_target and metric != "anls":
                    # In the case where we have multiple targets, return true if any are true
                    # TODO this may break for multipLe_target, non zero-or-1 metrics
                    scores = []
                    if not isinstance(gold, list):
                        # Sometimes, a multiple_target dataset has exceptions where one doc has
                        # only one string answer.
                        gold = [gold]

                    for gold_option in gold:
                        try:
                            result_score = self._metric_fn_list[metric](
                                references=[gold_option],
                                predictions=result,
                                **self._metric_fn_kwargs[metric],
                            )
                        except TypeError:  # TODO this is hacky and I don't want to do it
                            result_score = self._metric_fn_list[metric]([gold_option, result])

                        if isinstance(result_score, dict):
                            # TODO this handles the case where HF evaluate returns a dict.
                            result_score = result_score[metric]
                        scores.append(result_score)
                    result_score = 1.0 if any(scores) else 0.0
                else:
                    if not isinstance(gold, list):
                        gold = [gold]
                    try:
                        result_score = self._metric_fn_list[metric](
                            references=gold,
                            predictions=result,
                            **self._metric_fn_kwargs[metric],
                        )
                    except TypeError:  # Needed for now to use our metrics and HF Evaluate metrics
                        result_score = self._metric_fn_list[metric]([gold, result])
                    if isinstance(result_score, dict):
                        # TODO this handles the case where HF evaluate returns a dict.
                        result_score = result_score[metric]
                result_dict[metric] = result_score
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of"
                " 'loglikelihood','generate_until', 'generate_until_multi_round', or"
                " 'multiple_choice'",
            )

        return result_dict

    def aggregation(self) -> dict:
        """Aggregate the results."""
        return self._aggregation_list

    def higher_is_better(self) -> dict:
        """Get the higher is better property of the metrics."""
        return self._higher_is_better

    def get_config(self, key: str) -> Any:  # noqa: ANN401
        """Get the config value given its key.

        Args:
        ----
            key (str): The key to identify the value to return.

        """
        return getattr(self._config, key, None)

    @property
    def task_name(self) -> Any:  # noqa: ANN401
        """Get the task name."""
        return getattr(self.config, "task", None)

    def __repr__(self) -> str:
        """Return a string representation of the task output."""
        return (
            f"ConfigurableTask(task_name={getattr(self.config, 'task', None)},"
            f"output_type={self.OUTPUT_TYPE},"
            f"num_fewshot={getattr(self.config, 'num_fewshot', None)},"
            f"num_samples={len(self.eval_docs)})"
        )


class TaskManager:
    """Manager to index and load tasks.

    Args:
    ----
        include_path (str, list): An additional path to be searched for tasks recursively.
            Defaults to None.
        include_defaults (bool): If set to false, default tasks (those in src/data/tasks/) are
            not indexed. Defaults to True.
        model_name (str): The name of the model. Defaults to None.

    """

    def __init__(
        self,
        include_path: str | list | None = None,
        include_defaults: bool = True,
        model_name: str | None = None,
    ) -> None:
        self.include_path = include_path
        self.model_name = model_name

        self._task_index = self.init_tasks(
            include_path=include_path, include_defaults=include_defaults
        )
        self._all_tasks = sorted(list(self._task_index.keys()))

        self._all_groups = sorted(
            [x for x in self._all_tasks if self._task_index[x]["type"] == "group"]
        )
        self._all_subtasks = sorted(
            [x for x in self._all_tasks if self._task_index[x]["type"] == "task"]
        )
        self._all_tags = sorted(
            [x for x in self._all_tasks if self._task_index[x]["type"] == "tag"]
        )

        self.task_group_map = collections.defaultdict(list)

    @property
    def all_tasks(self) -> list:
        """Get all tasks."""
        return self._all_tasks

    @property
    def all_groups(self) -> list:
        """Get all groups."""
        return self._all_groups

    @property
    def all_subtasks(self) -> list:
        """Get all subtasks."""
        return self._all_subtasks

    @property
    def all_tags(self) -> list:
        """Get all tags."""
        return self._all_tags

    @property
    def task_index(self) -> dict:
        """Get the task index."""
        return self._task_index

    def init_tasks(
        self, include_path: str | list | None = None, include_defaults: bool = True
    ) -> dict:
        """Initialize the tasks.

        Args:
        ----
            include_path (str | list, optional): An additional path to be searched for tasks
                recursively. Defaults to None.
            include_defaults (bool): If set to false, default tasks (those in src/data/tasks/)
                are not indexed. Defaults to True.

        """
        all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"] if include_defaults else []

        if include_path is not None:
            if isinstance(include_path, str):
                include_path = [include_path]
            all_paths.extend(include_path)

        task_index = {}
        for task_dir in all_paths:
            tasks = self._get_task_and_group(task_dir)
            task_index = {**tasks, **task_index}

        return task_index

    def list_all_tasks(
        self, list_groups: bool = True, list_tags: bool = True, list_subtasks: bool = True
    ) -> str:
        """List all tasks.

        Args:
        ----
            list_groups (bool, optional): Whether to list groups. Defaults to True.
            list_tags (bool, optional): Whether to list tags. Defaults to True.
            list_subtasks (bool, optional): Whether to list subtasks. Defaults to True.

        """
        from pytablewriter import MarkdownTableWriter

        def sanitize_path(path: str) -> str:
            """Sanitize the path.

            Args:
            ----
                path (str): The path to sanitize.

            """
            if "src/data/tasks/" in path:
                return "src/data/tasks/" + path.split("src/data/tasks/")[-1]
            else:
                return path

        group_table = MarkdownTableWriter()
        group_table.headers = ["Group", "Config Location"]
        gt_values = []
        for g in self.all_groups:
            path = self.task_index[g]["yaml_path"]
            path = "---" if path == -1 else sanitize_path(path)
            gt_values.append([g, path])
        group_table.value_matrix = gt_values

        tag_table = MarkdownTableWriter()
        tag_table.headers = ["Tag"]
        tag_table.value_matrix = [[t] for t in self.all_tags]

        subtask_table = MarkdownTableWriter()
        subtask_table.headers = ["Task", "Config Location", "Output Type"]
        st_values = []
        for t in self.all_subtasks:
            path = self.task_index[t]["yaml_path"]

            output_type = ""

            # Read the yaml file to determine the output type
            if path != -1:
                config = utils.load_yaml_config(path, mode="simple")
                if "output_type" in config:
                    output_type = config["output_type"]
                elif "include" in config:
                    # If no output type, check if there is an include with an output type
                    include_path = path.split("/")[:-1] + config["include"]
                    include_config = utils.load_yaml_config(include_path, mode="simple")
                    if "output_type" in include_config:
                        output_type = include_config["output_type"]

            path = "---" if path == -1 else sanitize_path(path)
            st_values.append([t, path, output_type])
        subtask_table.value_matrix = st_values

        result = "\n"
        if list_groups:
            result += group_table.dumps() + "\n\n"
        if list_tags:
            result += tag_table.dumps() + "\n\n"
        if list_subtasks:
            result += subtask_table.dumps() + "\n\n"
        return result

    def match_tasks(self, task_list: list) -> list:
        """Match selected tasks to the indexed tasks.

        Args:
        ----
            task_list (list): The list of tasks.

        """
        return utils.pattern_match(task_list, self.all_tasks)

    def _name_is_registered(self, name: str) -> bool:
        """Check if a name is registered.

        Args:
        ----
            name (str): The name to check.

        """
        return name in self.all_tasks

    def _name_is_task(self, name: str) -> bool:
        """Check if a name is a task.

        Args:
        ----
            name (str): The name to check.

        """
        return bool(self._name_is_registered(name) and self.task_index[name]["type"] == "task")

    def _name_is_tag(self, name: str) -> bool:
        """Check if a name is a tag.

        Args:
        ----
            name (str): The name to check.

        """
        return bool(self._name_is_registered(name) and self.task_index[name]["type"] == "tag")

    def _name_is_group(self, name: str) -> bool:
        """Check if a name is a group.

        Args:
        ----
            name (str): The name to check.

        """
        return bool(self._name_is_registered(name) and self.task_index[name]["type"] == "group")

    def _name_is_python_task(self, name: str) -> bool:
        """Check if a name is a python task.

        Args:
        ----
            name (str): The name to check.

        """
        return bool(
            self._name_is_registered(name) and self.task_index[name]["type"] == "python_task"
        )

    def _get_yaml_path(self, name: str) -> str:
        """Get a yaml path given a name.

        Args:
        ----
            name (str): The name of the task

        """
        if name not in self.task_index:
            raise ValueError(f"Task {name} not found in task index.")
        return self.task_index[name]["yaml_path"]

    def _get_config(self, name: str) -> dict:
        """Get a config given a name.

        Args:
        ----
            name (str): The name of the task

        """
        if name not in self.task_index:
            raise ValueError(f"Task {name} not found in task index.")
        yaml_path = self._get_yaml_path(name)
        if yaml_path == -1:
            return dict()

        return utils.load_yaml_config(yaml_path, mode="full")

    def _get_task_list(self, name: str) -> list:
        """Get a task list given a name.

        Args:
        ----
            name (str): The name of the task

        """
        if self._name_is_task(name):
            raise ValueError(f"Task {name} is not a group.")
        return self.task_index[name]["task"]

    def _load_individual_task_or_group(  # TODO refactor?
        self,
        name_or_config: str | dict | None = None,
        parent_name: str | None = None,
        update_config: dict | None = None,
    ) -> dict:
        """Load an individual task or group.

        Args:
        ----
            name_or_config (str, dict, optional): The name or config of the task or group. Defaults
                to None.
            parent_name (str, optional): The name of the parent group. Defaults to None.
            update_config (dict, optional): The update config. Defaults to None.

        """

        def _load_task(config: dict, task: str) -> dict:
            """Load a task.

            Args:
            ----
                config (dict): The config of the task.
                task (str): The name of the task.

            """
            if "include" in config:
                config = {
                    **utils.load_yaml_config(
                        yaml_path=None,
                        yaml_config={"include": config.pop("include")},
                        mode="full",
                    ),
                    **config,
                }
            if _config_is_python_task(config):
                if _class_has_config_in_constructor(config["class"]):
                    task_object = config["class"](config=config)
                else:
                    task_object = config["class"]()
                if isinstance(task_object, ConfigurableTask):
                    # Very scuffed: set task name here. TODO fixme?
                    task_object.config.task = config["task"]
            else:
                task_object = ConfigurableTask(config=config, model_name=self.model_name)

            return {task: task_object}

        def _get_group_and_subtask_from_config(config: dict) -> tuple:
            """Get a group and subtask from a config.

            Args:
            ----
                config (dict): The config.

            """
            group_name = ConfigurableGroup(config=config)
            subtask_list = []
            for task in group_name.config["task"]:
                if isinstance(task, str) and self._name_is_tag(task):
                    subtask_list.extend(self._get_task_list(task))
                else:
                    subtask_list.append(task)
            return group_name, subtask_list

        def _process_group_config(config: dict, update_config: dict | None = None) -> tuple:
            """Process a group config.

            Args:
            ----
                config (dict): The config.
                update_config (dict, optional): The update config. Defaults to None.

            """
            if update_config is not None:
                config = {**config, **update_config}
            _update_config = {k: v for k, v in config.items() if k not in GROUP_ONLY_KEYS}
            if not bool(_update_config):
                _update_config = None

            group_config = {k: v for k, v in config.items() if k in GROUP_ONLY_KEYS}
            return group_config, _update_config

        if isinstance(name_or_config, str):
            if update_config is not None:
                # Process name_or_config as a dict instead
                name_or_config = {"task": name_or_config, **update_config}
            elif self._name_is_task(name_or_config) or self._name_is_python_task(name_or_config):
                task_config = self._get_config(name_or_config)
                return _load_task(task_config, task=name_or_config)
            else:
                subtask_list = self._get_task_list(name_or_config)
                if subtask_list == -1:
                    group_config = self._get_config(name_or_config)
                    group_config, update_config = _process_group_config(group_config)
                    group_name, subtask_list = _get_group_and_subtask_from_config(group_config)
                else:
                    if self._name_is_tag(name_or_config):
                        fn = partial(
                            self._load_individual_task_or_group,
                            update_config=name_or_config
                            if isinstance(name_or_config, dict)
                            else None,
                        )
                        return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
                    else:
                        group_name = ConfigurableGroup(
                            config={"group": name_or_config, "task": subtask_list}
                        )

        if isinstance(name_or_config, dict):
            if _config_is_task(name_or_config):
                name = name_or_config.pop("task")
                if update_config is not None:
                    name_or_config = {**name_or_config, **update_config}
                # If the name is registered as a group
                if self._name_is_group(name):
                    group_config = self._get_config(name)

                    group_config, update_config = _process_group_config(
                        group_config, name_or_config
                    )
                    group_name, subtask_list = _get_group_and_subtask_from_config(group_config)
                elif self._name_is_tag(name):
                    subtask_list = self._get_task_list(name)
                    fn = partial(
                        self._load_individual_task_or_group,
                        update_config=name_or_config,
                    )
                    return dict(collections.ChainMap(*map(fn, reversed(subtask_list))))
                else:
                    if self._name_is_registered(name):
                        base_task_config = self._get_config(name)

                        # Check if this is a duplicate.
                        if parent_name is not None:
                            num_duplicate = len(
                                list(
                                    filter(
                                        lambda x: x.startswith(name),
                                        self.task_group_map[parent_name],
                                    )
                                )
                            )
                            if num_duplicate > 0:
                                name = f"{name}-{num_duplicate}"
                            self.task_group_map[parent_name].append(name)

                        task_config = {
                            **base_task_config,
                            **name_or_config,
                        }
                    else:
                        task_config = name_or_config
                    return _load_task(task_config, task=name)
            else:
                group_config, update_config = _process_group_config(name_or_config)
                group_name, subtask_list = _get_group_and_subtask_from_config(group_config)

        fn = partial(
            self._load_individual_task_or_group,
            parent_name=group_name,
            update_config=update_config,
        )
        return {group_name: dict(collections.ChainMap(*map(fn, reversed(subtask_list))))}

    def load_task_or_group(self, task_list: str | list | None = None) -> dict:
        """Load a dictionary of task objects from a list.

        Args:
        ----
            task_list (str, list, optional): The list of tasks. Defaults to None.

        """
        if isinstance(task_list, str):
            task_list = [task_list]

        all_loaded_tasks = dict(
            collections.ChainMap(*map(self._load_individual_task_or_group, task_list))
        )
        return all_loaded_tasks

    def load_config(self, config: dict) -> dict:
        """Load a config.

        Args:
        ----
            config (dict): The config.

        """
        return self._load_individual_task_or_group(config)

    def _get_task_and_group(self, task_dir: str) -> dict:
        """Create a dictionary of tasks index.

        The dictionary has the following metadata:
            - `type`, that can be either `task`, `python_task`, `group` or `tags`.
                `task` refer to regular task configs, `python_task` are special
                yaml files that only consists of `task` and `class` parameters.
                `group` are group configs. `tags` are labels that can be assigned
                to tasks to assist in sorting and calling tasks of certain themes.
            - `yaml_path`, path to the yaml file. If the entry is a `group` that
                was configured through a task config, the yaml_path will be -1
                and all subtasks will be listed in `task` (see below)
            - `task`, reserved for entries with `type` as `group`. This will list
                all subtasks. When a group config is created (as opposed to task
                config having `group` parameter set), this will be set to -1 to
                avoid recursive indexing. The whole list of subtasks will be loaded
                at evaluation.

        Args:
        ----
            task_dir (str): The directory to check for tasks.

        """
        print_info = True
        ignore_dirs = [
            "__pycache__",
            ".ipynb_checkpoints",
        ]
        tasks_and_groups = collections.defaultdict()
        for root, dirs, file_list in os.walk(task_dir):
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    config = utils.load_yaml_config(yaml_path, mode="simple")
                    if _config_is_python_task(config):
                        # This is a python class config
                        tasks_and_groups[config["task"]] = {
                            "type": "python_task",
                            "yaml_path": yaml_path,
                        }
                    elif _config_is_group(config):
                        # This is a group config
                        tasks_and_groups[config["group"]] = {
                            "type": "group",
                            # This signals that we don't need to know the task list for indexing
                            # as it can be loaded when called.
                            "task": -1,
                            "yaml_path": yaml_path,
                        }

                    elif _config_is_task(config):
                        # This is a task config
                        task = config["task"]
                        tasks_and_groups[task] = {
                            "type": "task",
                            "yaml_path": yaml_path,
                        }

                        for attr in ["tag", "group"]:
                            if attr in config:
                                if attr == "group" and print_info:
                                    log.debug(
                                        "`group` and `group_alias` keys in tasks' configs will no"
                                        " longer be used in the next release of lmms-eval. `tag`"
                                        " will be used to allow to call a collection of tasks just"
                                        " like `group`. `group` will be removed in order to not"
                                        " cause confusion with the new ConfigurableGroup which "
                                        " will be the official way to create groups with addition"
                                        " of group-wide configurations."
                                    )
                                    print_info = False

                                attr_list = config[attr]
                                if isinstance(attr_list, str):
                                    attr_list = [attr_list]

                                for tag in attr_list:
                                    if tag not in tasks_and_groups:
                                        tasks_and_groups[tag] = {
                                            "type": "tag",
                                            "task": [task],
                                            "yaml_path": -1,
                                        }
                                    elif tasks_and_groups[tag]["type"] != "tag":
                                        log.warning(
                                            "The tag %s is already registered as a group, this tag"
                                            " will not be registered. This may affect tasks you"
                                            " want to call.",
                                            tag,
                                        )
                                        break
                                    else:
                                        tasks_and_groups[tag]["task"].append(task)
                    else:
                        log.debug("File %s in %s could not be loaded as a task or group", f, root)

        return tasks_and_groups
