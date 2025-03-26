from collections.abc import Callable

from src.models._base import Model
from src.schema import ModelInfo

__all__ = [
    "MODELS",
    "get_model",
    "get_model_builder",
    "get_model_info",
    "get_models_info",
    "register_model",
]

MODELS: dict[str, ModelInfo] = {}


def get_model(model_id: str, **model_kwargs) -> Model:
    """Get a model.

    Args:
    ----
        model_id (str): The name of the model.
        model_kwargs (dict): Keyword arguments to pass to the model builder.

    """
    return MODELS[model_id].builder_fn(**model_kwargs)


def get_model_builder(model_id: str) -> Callable | None:
    """Get a model builder.

    Args:
    ----
        model_id (str): The name of the model.

    """
    return MODELS[model_id].builder_fn


def get_model_info(model_id: str) -> ModelInfo:
    """Get the model info.

    Args:
    ----
        model_id (str): The name of the model.

    """
    return MODELS[model_id]


def get_models_info() -> list[ModelInfo]:
    """Get all models info."""
    return list(MODELS.values())


def register_model(name: str | None = None) -> Callable:
    """Register a model.

    Args:
    ----
        name (str, optional): The name of the model.

    """

    def decorator(model: Callable) -> Callable:
        MODELS[name or model.__name__.lower()] = ModelInfo(
            name=name or model.__name__.lower(),
            builder_fn=model,
        )
        return model

    return decorator
