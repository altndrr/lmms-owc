import abc
import inspect
import random
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import datasets
from datasets import Image, Sequence
from PIL import ImageFile
from tqdm import tqdm

from src import utils
from src.data.filters import get_filters_ensemble
from src.data.metrics import get_metric_builder, get_metric_info, get_metric_stderr_builder
from src.data.tasks._config import TaskConfig

__all__ = ["Task", "TaskInstance", "TaskOutput"]

log = utils.get_logger(__name__, rank_zero_only=True)

# HuggingfaceM4/NoCaps contains truncated image in test split
# Include this inside code block to avoid error
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class TaskInstance:
    """Dataclass describing a task instance."""

    request_type: str
    arguments: tuple
    idx: int
    metadata: dict = field(default_factory=dict)
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # Initialized after init
    task_name: str | None = None
    doc_id: str | None = None
    repeats: str | None = None
    doc: dict | None = None

    def __post_init__(self) -> None:
        """Post-init fields from unpacking metadata."""
        self.task_name = self.metadata["task"]
        self.doc_id = self.metadata["doc_id"]
        self.repeats = self.metadata["repeats"]

    @property
    def args(self) -> tuple:
        """Get the arguments of the task instance."""
        return self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)


class Task(abc.ABC):
    """An evaluation task.

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

    """

    VERSION = None
    DATASET_PATH: str = None  # The name as denoted in the HuggingFace datasets Hub
    DATASET_NAME: str = None  # The name of a subset within `DATASET_PATH`.
    OUTPUT_TYPE: str = None

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: datasets.DownloadMode | None = None,
        config: dict | None = None,
    ) -> None:
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self._instances = None

        self._config = TaskConfig(**config) if config else TaskConfig()

        self._filters = [get_filters_ensemble("none", [("take_first", None)])]

    def download(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: datasets.DownloadMode | None = None,
    ) -> None:
        """Download and returns the task dataset.

        Override this method to download the dataset from a custom API.

        Args:
        ----
            data_dir (str, optional): The path to a local folder containing the `Task`'s data
                files. Use this to specify the path to manually downloaded data (usually when the
                dataset is not publicly accessible). Defaults to None.
            cache_dir (str, optional): The directory to read/write the `Task` dataset. This follows
                the HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`. Defaults to None.
            download_mode (datasets.DownloadMode, optional): How to treat pre-existing `Task`
                downloads and data. Defaults to None.

        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )
        self.dataset_no_image = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

        for doc_name in self.dataset_no_image:
            remove_cols = []
            features = self.dataset_no_image[doc_name].features

            # Remove all Image instances from the dataset
            for feature_name in features:
                column = features[feature_name]
                is_image = isinstance(column, Image)
                is_image_seq = isinstance(column, Sequence) and isinstance(column.feature, Image)

                if is_image or is_image_seq:
                    remove_cols.append(feature_name)
            for remove_col in remove_cols:
                no_image_dataset = self.dataset_no_image[doc_name].remove_columns(remove_col)
                self.dataset_no_image[doc_name] = no_image_dataset

    @property
    def config(self) -> TaskConfig:
        """Return the TaskConfig associated with this class."""
        return self._config

    @abc.abstractmethod
    def has_training_docs(self) -> bool:
        """Whether the task has a training set."""
        raise NotImplementedError

    @abc.abstractmethod
    def has_validation_docs(self) -> bool:
        """Whether the task has a validation set."""
        raise NotImplementedError

    @abc.abstractmethod
    def has_test_docs(self) -> bool:
        """Whether the task has a test set."""
        raise NotImplementedError

    def training_docs(self) -> list:
        """Get the training docs for the `doc_to_text` method."""
        return []

    def validation_docs(self) -> list:
        """Get the validation docs for the `doc_to_text` method."""
        return []

    def validation_docs_no_media(self) -> datasets.Dataset | list:
        """Get the validation docs without media."""
        if self.has_validation_docs():
            return self.dataset_no_image[self.config.validation_split]

        return []

    def test_docs(self) -> list:
        """Get the test docs for the `doc_to_text` method."""
        return []

    def test_docs_no_media(self) -> datasets.Dataset | list:
        """Get the test docs without media."""
        if self.has_test_docs():
            return self.dataset_no_image[self.config.test_split]

        return []

    def fewshot_docs(self) -> list:
        """Get the fewshot docs for the `doc_to_text` method."""
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            if self.config.num_fewshot is not None:
                log.warning(
                    "`has_training_docs` and `has_validation_docs` are False,"
                    " using `test_docs` as `fewshot_docs`, but this is not recommended."
                )
            return self.test_docs()

    def _process_doc(self, doc: dict) -> dict:
        """Process (detokenize, strip, replace, etc.) individual documents.

        Override this method to process documents before they are passed to the model.
        This can be used in a map over documents of a data split.

        Args:
        ----
            doc (dict): A single document.

        """
        return doc

    @property
    def instances(self) -> list:
        """Get the list of dataset instances."""
        return self._instances

    def fewshot_examples(self, k: int, rnd: random.Random) -> list:
        """Get the fewshot examples for the `doc_to_text` method.

        Args:
        ----
            k (int): The number of examples to return.
            rnd (random.Random): The random number generator.

        """
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc: dict) -> str | None:
        """Get the decontamination query for the `doc_to_text` method.

        Override this method with specific decontamination query.

        Args:
        ----
            doc (dict): A single document.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_text(self, doc: dict) -> str:
        """Get the text for the `doc_to_text` method.

        Args:
        ----
            doc (dict): A single document.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_target(self, doc: dict) -> int | str | list:
        """Get the target for the `doc_to_text` method.

        Args:
        ----
            doc (dict): A single document.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def doc_to_visual(self, doc: dict) -> int | str | list:
        """Get the target for the `doc_to_visual` method.

        Args:
        ----
            doc (dict): A single document.

        """
        raise NotImplementedError

    def build_all_requests(
        self,
        *,
        limit: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable | None = None,
        tokenizer_name: str = "",
    ) -> None:
        """Build a set of Instances for a task, and store them in task.instances.

        Args:
        ----
            limit (int): The maximum number of instances to return. Defaults to None.
            rank (int): The rank of the current process. Defaults to 0.
            world_size (int): The number of processes. Defaults to 1.
            cache_requests (bool): Whether to cache requests. Defaults to False.
            rewrite_requests_cache (bool): Whether to rewrite the requests cache. Defaults to
                False.
            system_instruction (str): The system instruction. Defaults to None.
            apply_chat_template (bool): Whether to apply the chat template. Defaults to False.
            fewshot_as_multiturn (bool): Whether to use fewshot as multi-turn. Defaults to False.
            chat_template (Callable): The chat template. Defaults to None.
            tokenizer_name (str): The name of the tokenizer. Defaults to "".

        """
        if self.has_test_docs():
            _ = self.test_docs()
            split = self.config.test_split
        elif self.has_validation_docs():
            _ = self.validation_docs()
            split = self.config.validation_split
        else:
            raise ValueError("No test or validation docs found")

        # Used with caching
        og_limit = limit

        cache_key = f"requests-{self._config.task}-{self.config.num_fewshot}shot-rank{rank}-world_size{world_size}"  # noqa: E501
        cache_key += "-chat_template" if apply_chat_template else ""
        cache_key += "-fewshot_as_multiturn" if fewshot_as_multiturn else ""
        cache_key += (
            f"-system_prompt_hash{utils.hash_string(system_instruction)}"
            if system_instruction is not None
            else ""
        )
        cache_key += f"-tokenizer{tokenizer_name}"

        cached_instances = utils.load_from_cache(file_name=cache_key)

        if cache_requests and cached_instances and not rewrite_requests_cache:
            cached_instances = cached_instances[:limit]
            flattened_instances = [
                instance for instance_group in cached_instances for instance in instance_group
            ]
            self._instances = flattened_instances
            return

        log.info("Building contexts for %s on rank %d...", self.config.task, rank)

        instances = []

        # Process all documents when caching is specified for simplicity
        is_caching_specified = cache_requests and (not cached_instances or rewrite_requests_cache)
        if is_caching_specified and limit is not None:
            limit = None

        doc_id_docs = utils.create_iterator(
            enumerate(self.eval_docs_no_media),
            rank=rank,
            limit=int(limit) if limit else None,
            world_size=world_size,
        )
        doc_iterator_for_counting = (
            islice(range(len(self.test_docs())), rank, limit, world_size)
            if self.has_test_docs()
            else islice(range(len(self.validation_docs())), rank, limit, world_size)
        )
        num_docs = sum(1 for _ in doc_iterator_for_counting)

        for doc_id, doc in tqdm(doc_id_docs, total=num_docs):
            # Sample fewshot context
            # TODO need to offset doc_id by rank
            fewshot_ctx = self.fewshot_context(
                doc,
                0 if self.config.num_fewshot is None else self.config.num_fewshot,
                system_instruction,
                apply_chat_template,
                fewshot_as_multiturn,
                chat_template,
            )

            # TODO should override self.config.repeats if doing greedy to save time and compute
            per_task_metadata = {
                "task": self.config["task"],
                "doc_id": doc_id,
                "repeats": self.config.repeats,
                "split": split,
            }
            # TODO temporary fix for metadata loading, ignore the list of dict type.
            if self.config.metadata and isinstance(self.config.metadata, dict):
                per_task_metadata.update(self.config.metadata)

            inst = self.construct_requests(
                doc_id=doc_id, ctx=fewshot_ctx, metadata=per_task_metadata
            )

            if not isinstance(inst, list):
                inst = [inst]

            instances.append(inst)

        # Flatten, this is to allow slicing to work with pickles
        sliced_instances = instances[:og_limit]
        flattened_instances = [
            instance for instance_group in sliced_instances for instance in instance_group
        ]

        self._instances = flattened_instances

        if len(self._instances) == 0:
            raise ValueError("task.build_requests() did not find any docs!")

        if cache_requests and (not cached_instances or rewrite_requests_cache):
            utils.save_to_cache(file_name=cache_key, obj=instances)

        # ! Need to check if the `doc_to_visual` exists and restore it if so.
        # ! If we use cache, the doc_to_visual will be None since it's not serializable
        for instance in self._instances:
            if instance.arguments[2] is None:
                arguments = (
                    instance.arguments[0],
                    instance.arguments[1],
                    self.doc_to_visual,
                    *instance.arguments[3:],
                )
            else:
                arguments = instance.arguments

            instance.arguments = arguments

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def process_results(self, doc: dict, results: dict, **kwargs) -> dict:
        """Evaluate the prediction of an LMM on the input.

        It takes in the document, the results of the requests, and any other relevant information
        and returns a dictionary where keys are the names of sub-metrics and values are the
        values of the metric for that one document.

        Args:
        ----
            doc (dict): The document as returned from training_docs, validation_docs, or test_docs.
            results (dict): The results of the requests created in construct_requests.
            kwargs (dict): A dictionary of additional data, such as the LMM generation.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def aggregation(self) -> dict:
        """Aggregate the results."""
        raise NotImplementedError

    @abc.abstractmethod
    def higher_is_better(self) -> dict:
        """Get the higher is better property of the metrics."""
        raise NotImplementedError

    @utils.deprecated_positional
    def fewshot_context(
        self,
        doc_id: int,
        num_fewshot: int,
        split: str,
        rnd: random.Random | None = None,
        description: str | None = None,
    ) -> str:
        """Generate the fewshot context string.

        The context is made up of a prepended description (if provided), the `num_fewshot` number
        of examples, and an appended prompt example.

        Args:
        ----
            doc_id (int): The document id as returned from training_docs, validation_docs, or
                test_docs.
            num_fewshot (int): The number of fewshot examples to provide in the returned context
                string.
            split (str): The split of the document to retrieve from the dataset
            rnd (random.Random): The pseudo-random number generator used to randomly sample
                examples. Defaults to None.
            description (str): The task's description that will be prepended to the fewshot
                examples. Defaults to None.

        """
        if rnd is None:
            raise ValueError("A `random.Random` generator argument must be provided to `rnd`")

        description = description if description else ""
        doc = self.dataset_no_image[split][doc_id]

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # For sets without training docs, draw from other set while ensuring no overlap
            if self.has_training_docs():
                fewshot_examples = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshot_examples = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshot_examples = [x for x in fewshot_examples if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshot_examples]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def apply_filters(self) -> list | None:
        """Apply filters to the instances."""
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances, None)
        else:
            log.warning("No filter defined, passing through instances")
            return self._instances

    def dump_config(self) -> dict:
        """Dump the task's configuration as a dictionary."""
        # TODO should only return the overrides applied to a non-YAML task's configuration
        return self.config.to_dict()

    def set_config(self, key: str, value: Any, update: bool = False) -> None:  # noqa: ANN401
        """Set or update the configuration for a given key.

        Args:
        ----
            key (str): The key to set or update.
            value (Any): The value to set or update.
            update (bool, optional): If True, update the value if the key already exists.
                Defaults to False.

        """
        if key is None:
            raise ValueError("Key must be provided.")

        if update:
            current_value = getattr(self._config, key, {})
            if not isinstance(current_value, dict):
                raise TypeError(
                    f"Expected a dict for key '{key}', got {type(current_value).__name__} instead."
                )
            current_value.update(value)
        else:
            setattr(self._config, key, value)

    def override_metric(self, metric_name: str) -> None:
        """Override the default metrics used for evaluation with custom metrics.

        Args:
        ----
            metric_name (str): The name of the custom metric to override. Should be registered in
                src.data.metrics.

        """
        from src.data.tasks._manager import ConfigurableTask

        self._metric_fn_list = {}
        self._aggregation_list = {}
        self._metric_fn_kwargs = {}
        self._higher_is_better = {}

        metric_info = get_metric_info(metric_name)
        self._metric_fn_list[metric_name] = metric_info.builder_fn
        self._aggregation_list[metric_name] = metric_info.group_fn
        self._higher_is_better[metric_name] = metric_info.higher_is_better
        self._metric_fn_kwargs[metric_name] = {}

        if not isinstance(self, ConfigurableTask):
            self.process_results = lambda x, y: {metric_name: get_metric_builder(metric_name)}
            self.aggregation = lambda: {metric_name: get_metric_info(metric_name).group_fn}
        self._config.metric_list = [{"metric": metric_name}]
        self._config.process_results = None

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        """Set the seed for the random number generator used for fewshot examples.

        Args:
        ----
            seed (int, optional): The seed for the random number generator. Defaults to None.

        """
        self.fewshot_rnd = random.Random(seed)  # noqa: S311
        if hasattr(self, "sampler"):
            self.sampler.rnd = self.fewshot_rnd

    @property
    def eval_docs(self) -> datasets.Dataset | list[dict]:
        """Get the evaluation documents."""
        if self.has_test_docs():
            return self.test_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) has no valid"
                " `validation_docs` or `test docs`!"
            )

    @property
    def eval_docs_no_media(self) -> datasets.Dataset | list[dict]:
        """Get the evaluation docs without media."""
        if self.has_test_docs():
            return self.test_docs_no_media()
        elif self.has_validation_docs():
            return self.validation_docs_no_media()
        else:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) has no valid"
                " `validation_docs` or `test docs`!"
            )

    def doc_iterator(
        self, *, rank: int = 0, limit: int | None = None, world_size: int = 1
    ) -> Iterator[tuple[int, Any]]:
        """Get an iterator over the evaluation documents.

        Args:
        ----
            rank (int, optional): The rank of the current process. Defaults to 0.
            limit (int, optional): The maximum number of documents to return. Defaults to None.
            world_size (int, optional): The total number of processes. Defaults to 1.

        """
        limit = int(limit) if limit else None
        doc_iterator = utils.create_iterator(
            enumerate(self.eval_docs),
            rank=int(rank),
            limit=limit,
            world_size=int(world_size),
        )
        return doc_iterator


class TaskOutput:
    """Class for storing task outputs."""

    def __init__(
        self,
        task: Task | None = None,
        task_name: str | None = None,
        task_config: dict | None = None,
        version: str | None = None,
        group_name: str | None = None,
        n_shot: int | None = None,
        task_alias: str | None = None,
        group_alias: str | None = None,
        is_group: bool | None = None,
    ) -> None:
        self.task = task
        self.task_config = task_config
        self.task_name = task_name
        self.group_name = group_name
        self.version = version
        self.n_shot = n_shot
        self.task_alias = task_alias
        self.group_alias = group_alias
        self.is_group = is_group
        self.logged_samples = []
        self.sample_len = None
        self.sample_metrics = defaultdict(list)
        self.agg_metrics = defaultdict(list)
        self.args = None

    @classmethod
    def from_task_dict(cls, task_name: str, task: Task) -> "TaskOutput":
        """Create a TaskOutput from a task dict.

        Args:
        ----
            task_name (str): The name of the task.
            task (tuple | dict): The task.

        """
        if isinstance(task, tuple):
            group_name, task = task
        else:
            group_name = None

        if not task:
            # These gets filtered out in get_tasks_as_list once they are added to group hierarchy
            return cls(task=task, task_name=task_name, is_group=True, group_name=group_name)

        version = task.VERSION
        task_config = dict(task.dump_config())
        if (n_shot := task_config.get("num_fewshot")) == 0:
            meta_config = task_config.get("metadata", {})
            if isinstance(meta_config, dict):
                n_shot = meta_config.get("num_fewshot", 0)
            else:
                log.info(
                    "No metadata found in task config for %s, using default n_shot=0",
                    task_name,
                )
                n_shot = 0
        task_alias = task_config.get("alias")
        group_alias = task_config.get("group_alias")
        return cls(
            task=task,
            task_name=task_name,
            task_config=task_config,
            group_name=group_name,
            version=version,
            n_shot=n_shot,
            task_alias=task_alias,
            group_alias=group_alias,
        )

    def calculate_aggregate_metric(self, bootstrap_iters: int = 100000) -> None:
        """Calculate aggregate metrics for the task output.

        Args:
        ----
            bootstrap_iters (int, optional): The number of bootstrap iterations to use.
                Defaults to 100000.

        """
        for (metric, filter_key), items in self.sample_metrics.items():
            if metric in self.task.aggregation():
                agg_fn = self.task.aggregation()[metric]
                metric_key = f"{metric},{filter_key}"
                if "args" in inspect.signature(agg_fn).parameters:
                    self.agg_metrics[metric_key] = agg_fn(items, args=self.task.args)
                else:
                    self.agg_metrics[metric_key] = agg_fn(items)
                self.sample_len = len(items)  # TODO same sample size for each metric?
                if isinstance(bootstrap_iters, int):
                    stderr_fn = get_metric_stderr_builder(
                        metric=agg_fn,
                        bootstrap_iters=min(bootstrap_iters, 100)
                        if metric in ["bleu", "chrf", "ter"]
                        else bootstrap_iters,
                    )
                    self.agg_metrics[f"{metric}_stderr,{filter_key}"] = (
                        stderr_fn(items) if (stderr_fn and len(items) > 1) else "N/A"
                    )
                else:
                    raise ValueError(
                        f"Received bootstrap_iters '{bootstrap_iters}' but expected an integer."
                        " Set to 0 to turn off stderr calculations."
                    )

    def __repr__(self) -> str:
        """Return a string representation of the task output."""
        return (
            f"TaskOutput(task_name={self.task_name}, "
            f"group_name={self.group_name}, "
            f"version={self.version}, "
            f"n_shot={self.n_shot}, "
            f"task_alias={self.task_alias}, "
            f"group_alias={self.group_alias})"
        )
