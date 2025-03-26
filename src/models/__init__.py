from collections.abc import Callable

from src.models._api import (
    MODELS,
    get_model,
    get_model_builder,
    get_model_info,
    get_models_info,
    register_model,
)
from src.models._base import Model
from src.models._idefics2 import Idefics2
from src.models._instructblip import InstructBLIP
from src.models._internvl2 import InternVL2
from src.models._llava_hf import LLaVA
from src.models._llava_onevision import LLaVAOnevision
from src.models._phi3v import Phi3v
from src.models._qwen2_vl import Qwen2VL

__all__ = [
    "MODELS",
    "Model",
    "Idefics2",
    "InstructBLIP",
    "InternVL2",
    "LLaVA",
    "LLaVAOnevision",
    "Phi3v",
    "Qwen2VL",
    "register_model",
    "get_model",
    "get_model_builder",
    "get_model_info",
    "get_models_info",
    "register_model",
]

MODEL_TYPES: dict[str, Callable] = {
    "idefics2": Idefics2,
    "instructblip": InstructBLIP,
    "internvl2": InternVL2,
    "llava": LLaVA,
    "llava-onevision": LLaVAOnevision,
    "phi3v": Phi3v,
    "qwen2-vl": Qwen2VL,
}


@register_model("custom-model")
def custom_model(model_type: str, model_name_or_path: str, **model_kwargs) -> Callable:
    model_cls = MODEL_TYPES.get(model_type)
    if model_cls is None:
        raise ValueError(f"Model type '{model_type}' not found.")

    model_instance = model_cls(model_name_or_path, **model_kwargs)
    return model_instance
