import hashlib
import json
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Literal, TypeVar

import accelerate
import torch
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from transformers import BitsAndBytesConfig, PretrainedConfig, PreTrainedModel

from src import utils
from src.data.tasks import TaskInstance

__all__ = ["Model"]

log = utils.get_logger(__name__, rank_zero_only=True)

T = TypeVar("T", bound="Model")


class CacheHook:
    """A hook for caching model responses.

    Args:
    ----
        model (T | None): The model instance to attach the cache hook to. If None, caching is
            disabled.

    """

    def __init__(self, model: T | None) -> None:
        self.dbdict = getattr(model, "dbdict", None)

    def add_partial(self, attr: str, req: tuple, res: Any) -> None:  # noqa: ANN401
        """Add a response to the cache.

        Note: if caching is disabled (dbdict is None), this method returns without doing anything.
        The cache key is created by hashing the attribute name and request arguments combined.

        Args:
        ----
            attr (str): The attribute or method name associated with the request.
            req (tuple): The request arguments as a tuple.
            res (Any): The response to cache.

        """
        if self.dbdict is None:
            return

        hsh = hashlib.sha256(json.dumps([attr] + list(req)).encode("utf-8")).hexdigest()
        self.dbdict[hsh] = res


class Model(ABC):
    """Base class for all multi-modal models.

    It defines the interface that should be implemented by all Large Multimodal Models subclasses.
    Large Multimodal Models are assumed to take image-text as input and yield strings as output
    (inputs/outputs should be tokenization-agnostic.)

    Args:
    ----
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        distributed_types (list): List of supported distributed types. Default to None.
        kwargs: Additional keyword arguments.

    """

    accelerator: accelerate.Accelerator

    _model: PreTrainedModel | None = None
    _processor: Any | None = None
    _tokenizer: Any | None = None

    _device_map: str | None = None
    _dtype: torch.dtype | None = None
    _load_in_8bit: bool = False
    _load_in_4bit: bool = False
    _distributed_types: list | None = None
    _quantization_config: Any | None = None
    _rank: int = 0
    _world_size: int = 1

    def __init__(
        self,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        distributed_types: list[Literal["FSDP", "MULTI_GPU", "DEEPSPEED"]] | None = None,
        **kwargs,
    ) -> None:
        if len(kwargs.keys()) > 0:
            raise ValueError("kwargs are currently unsupported and unused in models.")

        if batch_size != 1:
            raise ValueError("Models currently only supports `batch_size=1`")

        if distributed_types is None:
            raise ValueError("`distributed_types` must be passed to the base Model constructor!")

        for distributed_type in distributed_types:
            if distributed_type not in DistributedType:
                raise KeyError(f"Invalid distributed type {distributed_type} passed!")

        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)

        if load_in_8bit or load_in_4bit:
            self._quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=dtype,
            )

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        self.accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.accelerator.num_processes > 1:
            device = f"cuda:{self.accelerator.local_process_index}"
            device_map = f"cuda:{self.accelerator.local_process_index}"
        self._device = torch.device(device)
        self._device_map = device_map
        self._dtype = dtype
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._distributed_types = [DistributedType[key] for key in distributed_types]
        self.apply_chat_template = False
        self.batch_size_per_gpu = int(batch_size)
        self.cache_hook = CacheHook(None)
        self.chat_template = None
        self.task_dict = {}

        self.load_model()
        if self._model is None:
            raise ValueError("The `load_model` method must set the attribute `_model`!")

        # Setup the accelerator
        if self.accelerator.num_processes > 1:
            if self.accelerator.distributed_type not in self._distributed_types:
                raise ValueError(
                    "Unsupported distributed type provided. Supported types are %s.",
                    ", ".join([key.name for key in self._distributed_types]),
                )

            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config
            # before using the model. Also, you have to select zero stage 0 (equivalent to DDP) in
            # order to make the prepare model works. I tried to set different parameters in the
            # kwargs to let default zero 2 stage works, but it didn't work.
            if self.accelerator.distributed_type.name == "DEEPSPEED":
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * self.accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(
                    must_match=True, **kwargs
                )
                log.info(
                    "Detected that you are using DistributedType.DEEPSPEED. Make sure you run"
                    " `accelerate config` and set zero stage to 0"
                )
            if self.accelerator.distributed_type.name in ["FSDP", "DEEPSPEED"]:
                self._model = self.accelerator.prepare(self.model)
            else:
                self._model = self.accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                log.info("Using %d devices with data parallelism", self.accelerator.num_processes)
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        elif self.accelerator.num_processes == 1 and self.device_map == "auto":
            log.info("Using %d devices with pipeline parallelism", self.accelerator.num_processes)
            self._rank = 0
            self._world_size = 1
        else:
            log.info("Using single device: %s", self._device)
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.batch_size_per_gpu

    @property
    def cache_hook(self) -> CacheHook:
        """Get the cache hook."""
        return self._cache_hook

    @cache_hook.setter
    def cache_hook(self, cache_hook: CacheHook) -> None:
        """Set the cache hook.

        Args:
        ----
            cache_hook (CacheHook): The cache hook.

        """
        self._cache_hook = cache_hook

    @property
    def config(self) -> PretrainedConfig:
        """Return the model config."""
        return self.model.config

    @property
    def device(self) -> torch.device:
        """Get the model device."""
        return self.model.device

    @property
    def device_map(self) -> str:
        """Get the model device map."""
        return self._device_map

    @property
    def dtype(self) -> torch.dtype | str:
        """Get the model dtype."""
        return self._dtype

    @property
    def eot_token_id(self) -> int:
        """Get the end-of-text token id."""
        if self.tokenizer is None:
            return -1
        return self.tokenizer.eos_token_id

    @property
    def model(self) -> PreTrainedModel:
        """Return the unwrapped model."""
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)

        return self._model

    @property
    def processor(self) -> Any:  # noqa: ANN401
        """Return the model processor."""
        return self._processor

    @property
    def tokenizer(self) -> Any:  # noqa: ANN401
        """Return the model tokenizer."""
        return self._tokenizer

    @property
    def rank(self) -> int:
        """Get the rank.

        The rank is used in the case of parallelism. Hardcoded to ensure no errors arise using API
        models which do not support multi-device parallelism nor expect it.
        """
        return self._rank

    @property
    def world_size(self) -> int:
        """Get the world size.

        The world size is used in the case of parallelism. Hardcoded to ensure no errors arise
        using API models which do not support multi-device parallelism nor expect it.
        """
        return self._world_size

    def eval(self) -> None:
        """Set the module in evaluation mode."""
        self.model.eval()

    def train(self) -> None:
        """Set the module in training mode."""
        self.model.train()

    @abstractmethod
    def load_model(self) -> None:
        """Load the model in memory."""
        raise NotImplementedError

    @abstractmethod
    def loglikelihood(self, requests: list[TaskInstance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.

        Downstream tasks should attempt to use loglikelihood instead of other
        LMM calls whenever possible.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, continuation). The arguments are as follows:
                - context (str): Context string. Implementations of LMM must be able to handle an
                    empty context string.
                - continuation (str):  The continuation over which log likelihood will be
                    calculated. If there is a word boundary, the space should be in the
                    continuation, e.g., context="hello" continuation=" world" is correct.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

    @abstractmethod
    def generate_until(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError

    @abstractmethod
    def generate_until_multi_round(self, requests: list[TaskInstance]) -> list[str]:
        """Generate greedily until a stopping sequence.

        Args:
        ----
            requests (list[TaskInstance]): A list of TaskInstance objects, with property `args`
                which returns a tuple (context, until). The arguments are as follows:
                - context (str): Context string.
                - until (str): The stopping sequence. The model should generate until this
                    sequence is generated. If the stopping sequence is not generated, the
                    model should generate until the maximum length is reached.
                - visual_list (list[dict]): Visual input to the model. Can be None.

        """
        raise NotImplementedError
