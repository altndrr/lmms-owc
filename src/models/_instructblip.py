import os
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Image as ImageType
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

__all__ = ["InstructBLIP"]

log = utils.get_logger(__name__, rank_zero_only=True)


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


def _add_order_label(image: Image.Image, label: str, font_size: int = 40) -> Image.Image:
    """Add a label to the top-left corner of an image.

    Args:
    ----
        image (Image.Image): The PIL Image to add the label to.
        label (str): The text to add as a label. Defaults to None.
        font_size (int): Size of the font to use. Defaults to 40.

    """
    draw = ImageDraw.Draw(image)

    font_path = os.path.join(__file__, os.pardir, "arial.ttf")
    font = ImageFont.truetype(font_path, font_size)

    text_width = text_height = font_size
    label_background_margin = 10
    label_background_size = (
        text_width + 2 * label_background_margin,
        text_height + 2 * label_background_margin,
    )

    label_background_position = (0, 0)  # Top-left corner
    draw.rectangle(
        (
            label_background_position,
            (
                label_background_position[0] + label_background_size[0],
                label_background_position[1] + label_background_size[1],
            ),
        ),
        fill="white",
    )

    label_position = (label_background_margin, label_background_margin)
    draw.text(label_position, label, font=font, fill="black")

    return image


def _concatenate_images_horizontal(image_list: list[Image.Image]) -> Image.Image:
    """Concatenate a list of images horizontally into a single image.

    Args:
    ----
        image_list (list[Image.Image]): A list of PIL Images to concatenate horizontally.
            All images must have the same height.

    """
    widths, heights = zip(*(i.size for i in image_list), strict=True)
    total_width = sum(widths)
    max_height = max(heights)
    if not all(height == max_height for height in heights):
        raise ValueError("All images must have the same height for horizontal concatenation")
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in image_list:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def _concatenate_images_vertical(image_list: list[Image.Image]) -> Image.Image:
    """Concatenate a list of images vertically into a single image.

    Args:
    ----
        image_list (list[Image.Image]): A list of PIL Images to concatenate vertically.
            All images must have the same width.

    """
    # Concatenate images horizontally
    widths, heights = zip(*(i.size for i in image_list), strict=True)
    total_height = sum(heights)
    max_width = max(widths)

    if not all(width == max_width for width in widths):
        raise ValueError("All images must have the same width for vertical concatenation")

    new_im = Image.new("RGB", (max_width, total_height))
    y_offset = 0
    for im in image_list:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im


def _resize_image_height(image: ImageType, fixed_size: int = 1008) -> Image.Image:
    """Resize an image to a fixed height while maintaining aspect ratio.

    Args:
    ----
        image (ImageType): The PIL Image to resize
        fixed_size (int, optional): The target height in pixels. Defaults to 1008.

    """
    width, height = image.size
    new_size = int(width * fixed_size / height), fixed_size
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _resize_image_width(image: Image.Image, fixed_size: int = 1008) -> Image.Image:
    """Resize an image to a fixed width while maintaining aspect ratio.

    Args:
    ----
        image (Image.Image): The PIL Image to resize
        fixed_size (int, optional): The target width in pixels. Defaults to 1008.

    """
    # Resize image, maintaining aspect ratio
    width, height = image.size
    new_size = fixed_size, int(height * fixed_size / width)
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _process_images_horizontal(original_images: list[ImageType], size: int = 1008) -> ImageType:
    """Process a list of images by resizing them to a fixed height and adding order labels.

    Args:
    ----
        original_images (list[ImageType]): List of PIL Image objects to be processed
        size (int, optional): Fixed height in pixels for resizing. Defaults to 1008.

    """
    images = []
    for i, img in enumerate(original_images):
        img_resized = _resize_image_height(img, fixed_size=size)
        img_labeled = _add_order_label(img_resized, f"[{i+1}]")
        images.append(img_labeled)

    return _concatenate_images_horizontal(images)


def _process_images_vertical(original_images: list[ImageType], size: int = 1008) -> ImageType:
    """Process a list of images by resizing them to a fixed width and adding order labels.

    Args:
    ----
        original_images (list[ImageType]): List of PIL Image objects to be processed
        size (int, optional): Fixed width in pixels for resizing. Defaults to 1008.

    """
    images = []
    for i, img in enumerate(original_images):
        img_resized = _resize_image_width(img, fixed_size=size)
        img_labeled = _add_order_label(img_resized, f"[{i+1}]")
        images.append(img_labeled)

    return _concatenate_images_vertical(images)


def process_images(images: list[ImageType], size: int = 1008) -> ImageType:
    """Process a list of images by concatenating them in a way that minimizes aspect ratio.

    Args:
    ----
        images (list[ImageType]): List of PIL Image objects to be processed. Default: None.
        size (int): Size in pixels for resizing the images. Default: 1008.

    """
    concat_horizontal = _process_images_horizontal(images, size)
    concat_vertical = _process_images_vertical(images, size)

    hw, hh = concat_horizontal.size
    vw, vh = concat_vertical.size

    ha = hw / hh
    va = vh / vw

    if ha > va:
        return concat_vertical
    else:
        return concat_horizontal


class InstructBLIP(Model):
    """InstructBLIP model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models.  Defaults to "Salesforce/instructblip-vicuna-7b".
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "Salesforce/instructblip-vicuna-7b",
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

    def _tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        """Encode a string into tokens using the model's tokenizer.

        Args:
        ----
            string (str): The input string to encode
            left_truncate_len (int, optional): If provided, truncate the encoded tokens from the
                left to this length. Defaults to None.
            add_special_tokens (bool, optional): Whether to add special tokens during encoding. If
                None, defaults to False. Defaults to None.

        """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # Left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def _tok_decode(self, tokens: int | list[int]) -> str:
        """Decode token(s) to text string using the model's tokenizer.

        Args:
        ----
            tokens: A single token ID or list of token IDs to decode.

        """
        return self.tokenizer.decode(tokens)

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
        }
        processor_kwargs = {}

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        self._model = InstructBlipForConditionalGeneration.from_pretrained(
            self._model_name_or_path, **model_kwargs
        )
        self._processor = InstructBlipProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._tokenizer = self._processor.tokenizer
        self.model.tie_weights()

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

        def _collate(x: tuple[str, ...]) -> tuple[int, str]:
            """Group and sort requests by context length for efficient batching.

            The negative sign on len(tokens) sorts in descending order, which provides several
                advantages:
                - Time estimates will be overestimates rather than underestimates, which is more
                    useful for planning;
                - The first item in a batch determines the padded context length, simplifying
                    batching logic;
                - Makes automatic adaptive batches much easier to implement;
                - Any out-of-memory errors occur immediately rather than near the end.

            Args:
            ----
                x: A tuple containing the context string and other arguments

            """
            tokens = self._tok_encode(x[0])
            return -len(tokens), x[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size

        pbar_kwargs = dict(total=num_iters, disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk, strict=True)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = _flatten_list(visuals)

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self._tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

            context = contexts[0]
            if "<image>" in context:
                # InstructBLIP does not expect the <image> tag
                context = context.replace("<image>", "")

            # Set truncation equals true here, the max length for qformer tokenizer is 512
            # If not truncate, some questions will cause size mismatch
            # The transformer implementation can't handle multi images for BLIP
            # Concat it into one image
            if len(visuals) > 1:
                visuals = [process_images(visuals)]
            inputs = self.processor(
                images=visuals, text=context, return_tensors="pt", truncation=True
            ).to(self.device)

            gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            cont = self.model.generate(
                **inputs,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            text_outputs = text_outputs.replace(context + " ", "").strip()
            res.append(text_outputs)

            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

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

        log.warning("The model do not fully support multi-turn as it does not store history")

        def _collate(x: tuple[str, ...]) -> tuple[int, str]:
            """Group and sort requests by context length for efficient batching.

            The negative sign on len(tokens) sorts in descending order, which provides several
                advantages:
                - Time estimates will be overestimates rather than underestimates, which is more
                    useful for planning;
                - The first item in a batch determines the padded context length, simplifying
                    batching logic;
                - Makes automatic adaptive batches much easier to implement;
                - Any out-of-memory errors occur immediately rather than near the end.

            Args:
            ----
                x: A tuple containing the context string and other arguments

            """
            tokens = self._tok_encode(x[0])
            return -len(tokens), x[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (len(requests) + self.batch_size - 1) // self.batch_size

        pbar_kwargs = dict(total=num_iters, disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk in chunks:
            (
                contexts,
                all_gen_kwargs,
                doc_to_visual,
                doc_to_text,
                doc_id,
                task,
                split,
            ) = zip(*chunk, strict=True)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = _flatten_list(visuals)

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self._tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        "Expected `gen_kwargs['until']` to be of type Union[str,list] but got"
                        f" {type(until)}"
                    )

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

                        result = doc_to_text[0](
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

                context = contexts[0]
                if "<image>" in context:
                    # InstructBLIP does not expect the <image> tag
                    context = context.replace("<image>", "")

                if isinstance(visuals, tuple):
                    visuals = list(*visuals)

                # Set truncation equals true here, the max length for qformer tokenizer is 512
                # If not truncate, some questions will cause size mismatch
                # The transformer implementation can't handle multi images for BLIP
                # Concat it into one image
                if len(visuals) > 1:
                    visuals = [process_images(visuals)]
                inputs = self.processor(
                    images=visuals, text=context, return_tensors="pt", truncation=True
                ).to(self.device)

                gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1
                cont = self.model.generate(
                    **inputs,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                text_outputs = text_outputs.replace(context + " ", "").strip()

                round_idx += 1
                batched_round_results.append([text_outputs])

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("instructblip-vicuna-13b")
def instructblip_vicuna_13b(**model_kwargs) -> Model:
    """Load the InstructBLIP model with Vicuna 13B."""
    model_name_or_path = "Salesforce/instructblip-vicuna-13b"
    model = InstructBLIP(model_name_or_path, **model_kwargs)
    return model


@register_model("instructblip-vicuna-7b")
def instructblip_vicuna_7b(**model_kwargs) -> Model:
    """Load the InstructBLIP model with Vicuna 7B."""
    model_name_or_path = "Salesforce/instructblip-vicuna-7b"
    model = InstructBLIP(model_name_or_path, **model_kwargs)
    return model
