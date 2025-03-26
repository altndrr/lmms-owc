import math
import os
from typing import Any

import torch
import torchvision.transforms as T
from PIL.Image import Image as ImageType
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

__all__ = ["InternVL2"]

log = utils.get_logger(__name__, rank_zero_only=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_GEN_KWARGS = dict(
    num_beams=1,
    max_new_tokens=1024,
    do_sample=False,
)


def _flatten_list(input: list[list[Any]]) -> list[Any]:
    """Flatten a nested list into a single list.

    Args:
    ----
        input (list): A nested list containing elements to be flattened.

    """
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def _build_transform(
    input_size: int, norm_mean: tuple = IMAGENET_MEAN, norm_std: tuple = IMAGENET_STD
) -> T.Compose:
    """Build an image transformation pipeline for preprocessing images.

    The transformation pipeline includes:
    1. Converting the image to RGB format if not already
    2. Resizing the image to the specified input size
    3. Converting the image to a tensor
    4. Normalizing the image using ImageNet mean and standard deviation values

    Args:
    ----
        input_size (int): The target size for both height and width of the image
        norm_mean (tuple): Mean RGB values to use for normalization. Defaults to Imagenet
            statistics.
        norm_std (tuple): Std RGB values to use for normalization. Defaults to Imagenet
            statistics.

    """
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=norm_mean, std=norm_std),
        ]
    )
    return transform


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Find the closest aspect ratio from target ratios that best matches the input aspect ratio.

    It not only considers the closest numerical match to the aspect ratio, but also takes into
    account the resulting image area to ensure reasonable proportions.

    Args:
    ----
        aspect_ratio (float): The aspect ratio (width/height) of the input image.
        target_ratios (list[tuple[int, int]]): List of candidate aspect ratios as (width, height)
            tuples.
        width (int): Original image width in pixels.
        height (int): Original image height in pixels.
        image_size (int): Target size for the image.

    """
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def _dynamic_preprocess(
    image: ImageType,
    min_num: int = 1,
    max_num: int = 6,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[ImageType]:
    """Dynamically preprocess an image by splitting it into multiple patches.

    It resizes and splits the input image into a number of equally-sized patches, with the number
    of patches determined by the image's aspect ratio and the min_num/max_num constraints.
    Optionally adds a thumbnail of the entire image.

    It attempts to maintain the original image's aspect ratio while dividing it into a reasonable
    number of patches. The actual number of patches will be between min_num and max_num, inclusive.

    Args:
    ----
        image (PIL.Image.Image): The input image to be preprocessed
        min_num (int, optional): Minimum number of patches to generate. Defaults to 1.
        max_num (int, optional): Maximum number of patches to generate. Defaults to 6.
        image_size (int, optional): Size of each square patch in pixels. Defaults to 448.
        use_thumbnail (bool, optional): Whether to append a thumbnail of the full image. Defaults
            to False.

    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio to the target
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # Split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _load_image(
    image: ImageType | str,
    input_size: int = 448,
    max_num: int = 6,
) -> torch.Tensor:
    """Load and preprocess an image for model input.

    It takes either a PIL Image or a path to an image file, processes it by resizing and splitting
    it into patches, and converts it into a tensor suitable for model input.

    It applies standard ImageNet normalization to the image tensors and includes a thumbnail of the
    full image in the output if more than one patch is generated.

    Args:
    ----
        image (Union[PIL.Image.Image, str]): Either a PIL Image object or a string path
            to an image file.
        input_size (int, optional): The target size for image patches in pixels.
            Defaults to 448.
        max_num (int, optional): Maximum number of patches to generate from the image.
            Defaults to 6.

    """
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)

    return pixel_values


def _split_model(model_name: str, num_layers: int | None = None) -> dict[str, int]:
    """Create a device map for model parallel loading across multiple GPUs.

    It generates a device mapping strategy that ensures the first and last layers of the model are
    on the same device (GPU 0) to prevent errors during multi-GPU inference caused by tensors being
    on different devices. The remaining layers are distributed across available GPUs.

    It assumes GPU 0 will be used for the vision model components, so it's treated as having only
    half its capacity available for language model layers.

    Args:
    ----
        model_name (str): The name of the model to create the device map for. Must be one of the
            supported InternVL2 model variants.
        num_layers (int | None, optional): The number of layers in the model. If None, it will be
            determined based on the model name. Defaults to None.

    """
    device_map = {}
    world_size = torch.cuda.device_count()
    if num_layers is None:
        num_layers = {
            "InternVL2_5-1B": 24,
            "InternVL2_5-2B": 24,
            "InternVL2_5-4B": 36,
            "InternVL2_5-8B": 32,
            "InternVL2_5-26B": 48,
            "InternVL2_5-38B": 64,
            "InternVL2_5-78B": 80,
            "InternVL2-1B": 24,
            "InternVL2-2B": 24,
            "InternVL2-4B": 32,
            "InternVL2-8B": 32,
            "InternVL2-26B": 48,
            "InternVL2-40B": 60,
            "InternVL2-Llama3-76B": 80,
        }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


class InternVL2(Model):
    """InternVL2 model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "OpenGVLab/InternVL2-2B".
        num_layers (int, optional): The number of layers. Defaults to None.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "OpenGVLab/InternVL2-2B",
        num_layers: int | None = None,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._num_layers = num_layers

        if load_in_8bit:
            log.warning("InternVL2 does not work with 8bit quantization, skipping it...")
            load_in_8bit = False
        if load_in_4bit:
            log.warning("InternVL2 does not work with 4bit quantization, skipping it...")
            load_in_4bit = False

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        # Overwrite the device and device mapping to split the model on multiple devices.
        if self.accelerator.num_processes > 1:
            self._device_map = f"cuda:{self.accelerator.local_process_index}"
        elif self.accelerator.num_processes == 1 and self.device_map == "auto":
            self._device_map = _split_model(
                self._model_name_or_path.split("/")[-1], num_layers=self._num_layers
            )
        else:
            self._device_map = f"cuda:{self.accelerator.local_process_index}"

        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "device_map": self.device_map,
        }
        tokenizer_kwargs = {
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "device_map": self.device_map,
        }

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        self._model = AutoModel.from_pretrained(self._model_name_or_path, **model_kwargs)
        self._processor = None
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path, **tokenizer_kwargs
        )

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
        res = []
        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)

        reg_args = [reg.args for reg in requests]
        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in reg_args:
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, _ in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = _flatten_list(visuals)
            if visuals:
                visuals = [_load_image(visual).to(self.dtype).cuda() for visual in visuals]
                pixel_values = torch.cat(visuals, dim=0)
                num_patches_list = [visual.size(0) for visual in visuals]
                image_tokens = ["<image>"] * len(visuals)
                image_tokens = " ".join(image_tokens)
                contexts = image_tokens + "\n" + contexts
            else:
                pixel_values = None
                num_patches_list = None
            response, history = self.model.chat(
                self.tokenizer,
                pixel_values,
                contexts,
                gen_kwargs,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
            res.append(response)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), response)
            pbar.update(1)

        pbar.close()
        return res

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
        res = []
        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)

        reg_args = [reg.args for reg in requests]
        for (
            contexts,
            gen_kwargs,
            doc_to_visual,
            doc_to_text,
            doc_id,
            task,
            split,
        ) in reg_args:
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = _flatten_list(visuals)

            if isinstance(doc_id, int):
                doc_id = [doc_id]

            if "until" in gen_kwargs:
                gen_kwargs.pop("until")
            for k, v in DEFAULT_GEN_KWARGS.items():
                if k not in gen_kwargs:
                    gen_kwargs[k] = v

            pop_keys = []
            for k, _ in gen_kwargs.items():
                if k not in DEFAULT_GEN_KWARGS:
                    pop_keys.append(k)

            for k in pop_keys:
                gen_kwargs.pop(k)

            # Terminate when receiving signal from the doc_to_text
            round_idx = 0
            batched_round_results, batched_round_info = [], []
            while True:
                last_round_info = None

                # Get current round visual and context from doc_to_text function
                if round_idx != 0:
                    results = []
                    for ids_idx, ids in enumerate(doc_id):
                        previous_round_results = [
                            round_results[ids_idx] for round_results in batched_round_results
                        ]
                        if len(batched_round_info) > 0:
                            last_round_info = batched_round_info[-1][ids_idx]

                        result = doc_to_text(
                            self.task_dict[task][split][ids],
                            round_idx=round_idx,
                            previous_round_results=previous_round_results,
                            last_round_info=last_round_info,
                        )
                        results.append(result)

                    (
                        visuals,
                        contexts,
                        batched_terminal_signal,
                        batched_round_results,
                        last_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_results = list(zip(*batched_round_results, strict=True))
                    last_round_info = last_round_info[-1]
                    if batched_terminal_signal[0]:  # terminal signal from doc_to_text function
                        break

                if isinstance(contexts, str):
                    contexts = [contexts]

                context = contexts[0]

                if isinstance(visuals, tuple):
                    visuals = [visual for visual in visuals]
                if all(visual is None for visual in visuals):
                    visuals = None

                history = None
                if last_round_info and "history" in last_round_info:
                    history = last_round_info["history"][0]

                pixel_values, num_patches_list = None, None
                if visuals:
                    visuals = [_load_image(visual).to(self.dtype).cuda() for visual in visuals]
                    pixel_values = torch.cat(visuals, dim=0)
                    num_patches_list = [visual.size(0) for visual in visuals]
                    image_tokens = ["<image>"] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    context = image_tokens + "\n" + context
                elif last_round_info and "pixel_values" in last_round_info:
                    pixel_values = last_round_info["pixel_values"][0]
                    num_patches_list = last_round_info["num_patches_list"][0]

                response, history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    context,
                    gen_kwargs,
                    num_patches_list=num_patches_list,
                    history=history,
                    return_history=True,
                )

                round_idx += 1
                batched_round_results.append([response])
                batched_round_info.append(
                    [
                        dict(
                            history=[history],
                            num_patches_list=[num_patches_list],
                            pixel_values=[pixel_values],
                        )
                    ]
                )

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        pbar.close()
        return res


@register_model("internvl2-40b")
def internvl2_40b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 40B params."""
    model_name_or_path = "OpenGVLab/InternVL2-40B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2-26b")
def internvl2_26b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 26B params."""
    model_name_or_path = "OpenGVLab/InternVL2-26B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2-8b")
def internvl2_8b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 8B params."""
    model_name_or_path = "OpenGVLab/InternVL2-8B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2-4b")
def internvl2_4b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 4B params."""
    model_name_or_path = "OpenGVLab/InternVL2-4B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2-2b")
def internvl2_2b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 2B params."""
    model_name_or_path = "OpenGVLab/InternVL2-2B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2-1b")
def internvl2_1b(**model_kwargs) -> Model:
    """Load the InternVL2 model with 1B params."""
    model_name_or_path = "OpenGVLab/InternVL2-1B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-38b")
def internvl25_38b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 38B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-38B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-26b")
def internvl25_26b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 26B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-26B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-8b")
def internvl25_8b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 8B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-8B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-4b")
def internvl25_4b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 4B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-4B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-2b")
def internvl25_2b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 2B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-2B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model


@register_model("internvl2.5-1b")
def internvl25_1b(**model_kwargs) -> Model:
    """Load the InternVL2.5 model with 1B params."""
    model_name_or_path = "OpenGVLab/InternVL2_5-1B"
    model = InternVL2(model_name_or_path, **model_kwargs)
    return model
