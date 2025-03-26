import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field

__all__ = [
    "AggregationInfo",
    "DatasetInfo",
    "FilterInfo",
    "MetricInfo",
    "ModelInfo",
    "SamplerInfo",
]


class AggregationInfo(BaseModel):
    """Info about an available aggregation metric."""

    name: str = Field(
        description="Aggregation metric key.",
        examples=["perplexity"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the aggregation metric.",
        examples=["src.data.metrics._group.perplexity"],
        exclude=True,
        default=None,
    )
    can_bootstrap: bool = Field(
        description="Whether the metric can be bootstrapped to evaluate its standard error",
        examples=[False],
    )


class DatasetInfo(BaseModel):
    """Info about an available dataset."""

    name: str = Field(
        description="Dataset key.",
        examples=["ok_vqa"],
    )
    description: str = Field(
        description="Description of the dataset.",
        examples=["OK-VQA"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the dataset.",
        examples=["src.data.datasets._huggingface.ok_vqa"],
        exclude=True,
        default=None,
    )
    dataset_path: str = Field(
        description="Path to the dataset.",
        examples=["lmms-lab/OK-VQA"],
    )
    splits: list[str] = Field(
        description="List of available splits.",
        default_factory=list,
    )
    num_rows: list[int] = Field(
        description="Number of rows per split.",
        default_factory=list,
    )

    @computed_field
    @property
    def status(self) -> Literal["unknown", "unavailable", "available"]:
        """Get the status of the dataset."""
        if self.dataset_path is None:
            return "unknown"

        cache_dir = os.getenv("HF_HOME")
        hub_dir = Path(cache_dir) / "hub"
        if cache_dir is None or not hub_dir.exists():
            return "unknown"

        # Get all the folders in the cache directory
        dataset_name = self.dataset_path.split("/")[-1].replace("-", "").lower()
        folders = [f for f in hub_dir.iterdir() if f.is_dir()]
        for folder in folders:
            folder_name = folder.name.split("--")[-1].replace("-", "").lower()
            if dataset_name in folder_name:
                return "available"

        return "unavailable"


class FilterInfo(BaseModel):
    """Info about an available filter."""

    name: str = Field(
        description="Filter key.",
        examples=["multi_choice_regex"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the filter.",
        examples=["src.data.filters._extraction.multi_choice_regex"],
        exclude=True,
        default=None,
    )


class MetricInfo(BaseModel):
    """Info about an available metric."""

    name: str = Field(
        description="Metric key.",
        examples=["perplexity"],
    )
    higher_is_better: bool = Field(
        description="Whether higher values are better.",
        examples=[False],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the metric.",
        examples=["src.data.metrics._instance.perplexity"],
        exclude=True,
        default=None,
    )
    group_fn: Callable | None = Field(
        description="Function to build the aggregation function used by the metric.",
        examples=["src.data.metrics._group.perplexity"],
        exclude=True,
        default=None,
    )
    output_types: list[str] = Field(
        description="The output types of the metrics.",
        default_factory=list,
        examples=[["loglikelihood"]],
    )
    can_bootstrap: bool = Field(
        description="Whether the metric can be bootstrapped to evaluate its standard error",
        examples=[True],
    )


class ModelInfo(BaseModel):
    """Info about an available model."""

    name: str = Field(
        description="Model key.",
        examples=["llava_one_vision"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the model.",
        examples=["src.models.llava_one_vision.llava_onevision_qwen2_7b"],
        exclude=True,
        default=None,
    )


class SamplerInfo(BaseModel):
    """Info about an available sampler."""

    name: str = Field(
        description="Sampler key.",
        examples=["first_n"],
    )
    builder_fn: Callable | None = Field(
        description="Function to build the sampler.",
        examples=["src.data.samplers._context.FirstNSampler"],
        exclude=True,
        default=None,
    )
