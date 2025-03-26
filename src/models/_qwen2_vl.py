import base64
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

__all__ = ["Qwen2VL"]

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


class Qwen2VL(Model):
    """Qwen2VL Model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether to use flash attention 2. Default to False.
        max_pixels (int): The max number of pixels in an image. Defaults to 12'845'056.
        min_pixels (int): The min number of pixels in an image. Defaults to 3'316.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://github.com/QwenLM/Qwen2-VL

    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct",
        use_cache: bool = True,
        use_flash_attention_2: bool | None = False,
        max_pixels: int = 1024 * 28 * 28,
        min_pixels: int = 4 * 28 * 28,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._use_cache = use_cache
        self._use_flash_attention_2 = use_flash_attention_2
        self._max_pixels = max_pixels
        self._min_pixels = min_pixels

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU"],
            **kwargs,
        )

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
        }
        processor_kwargs = {
            "max_pixels": self._max_pixels,
            "min_pixels": self._min_pixels,
        }

        if self._use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        PretrainedModel = Qwen2VLForConditionalGeneration
        if "Qwen2.5" in self._model_name_or_path:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
            except ImportError as e:
                raise ValueError(
                    "Failed to import Qwen2_5_VLForConditionalGeneration."
                    " Please upgrade transformers to a later version."
                ) from e

            PretrainedModel = Qwen2_5_VLForConditionalGeneration

        self._model = PretrainedModel.from_pretrained(self._model_name_or_path, **model_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

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
            tokens = self.tokenizer.encode(x[0])
            return -len(tokens), x[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [
                batched_doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in batched_doc_id
            ]
            batched_visuals = _flatten_list(batched_visuals)

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

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

            if isinstance(batched_contexts, tuple):
                batched_contexts = list(batched_contexts)

            for i in range(len(batched_contexts)):
                if "<image>" in batched_contexts[i]:
                    batched_contexts[i] = batched_contexts[i].replace("<image>", "")

            messages = []
            for i, context in enumerate(batched_contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": "You are a helpful assistant."}]

                if len(batched_visuals) > 0:
                    visual = batched_visuals[i] if i < len(batched_visuals) else None
                    if isinstance(visual, Image.Image):  # Single image
                        base64_image = visual.convert("RGB")
                        buffer = BytesIO()
                        base64_image.save(buffer, format="JPEG")
                        base64_bytes = base64.b64encode(buffer.getvalue())
                        base64_string = base64_bytes.decode("utf-8")
                        message.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "image": f"data:image/jpeg;base64,{base64_string}",
                                    },
                                    {"type": "text", "text": context},
                                ],
                            }
                        )
                    elif isinstance(visual, list | tuple) and all(
                        isinstance(v, Image.Image) for v in visual
                    ):  # Multiple images
                        image_content = []
                        for v in visual:
                            base64_image = v.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            image_content.append(
                                {
                                    "type": "image",
                                    "image": f"data:image/jpeg;base64,{base64_string}",
                                }
                            )
                        message.append(
                            {
                                "role": "user",
                                "content": image_content + [{"type": "text", "text": context}],
                            }
                        )
                    else:
                        message.append(
                            {"role": "user", "content": [{"type": "text", "text": context}]}
                        )
                else:
                    message.append(
                        {"role": "user", "content": [{"type": "text", "text": context}]}
                    )

                messages.append(message)

            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)

            if video_inputs is not None:
                raise ValueError(
                    "Video inputs should be empty for current implementation of Qwen2VL."
                )

            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id

            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=gen_kwargs["temperature"] > 0,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                use_cache=self._use_cache,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, cont, strict=True)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for ans, context in zip(answers, batched_contexts, strict=True):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
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
            tokens = self.tokenizer.encode(x[0])
            return -len(tokens), x[0]

        # Group requests by their generation_kwargs, so that we don't try to execute, e.g., greedy
        # sampling and temp=0.8 sampling in the same batch.
        reordered = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = reordered.get_batched(n=self.batch_size, batch_fn=None)

        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)
        for chunk in chunks:
            (
                batched_contexts,
                all_gen_kwargs,
                batched_doc_to_visual,
                batched_doc_to_text,
                batched_doc_id,
                batched_task,
                batched_split,
            ) = zip(*chunk, strict=True)
            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [
                batched_doc_to_visual[0](self.task_dict[task][split][ids])
                for ids in batched_doc_id
            ]
            batched_visuals = _flatten_list(batched_visuals)

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

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
                    for ids_idx, doc_id in enumerate(batched_doc_id):
                        previous_round_results = [
                            round_results[ids_idx] for round_results in batched_round_results
                        ]
                        if len(batched_round_info) > 0:
                            last_round_info = batched_round_info[-1][ids_idx]

                        result = batched_doc_to_text[0](
                            self.task_dict[task][split][doc_id],
                            round_idx=round_idx,
                            previous_round_results=previous_round_results,
                            last_round_info=last_round_info,
                        )
                        results.append(result)

                    (
                        batched_visuals,
                        batched_contexts,
                        batched_terminal_signal,
                        batched_round_results,
                        last_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_results = list(zip(*batched_round_results, strict=True))
                    last_round_info = last_round_info[-1]
                    if batched_terminal_signal[0]:  # terminal signal from doc_to_text function
                        break

                if isinstance(batched_contexts, tuple):
                    batched_contexts = list(batched_contexts)

                for i in range(len(batched_contexts)):
                    if "<image>" in batched_contexts[i]:
                        batched_contexts[i] = batched_contexts[i].replace("<image>", "")

                messages = []
                for i, context in enumerate(batched_contexts):
                    if "<image>" in context:
                        context = context.replace("<image>", "")

                    message = [{"role": "system", "content": "You are a helpful assistant."}]
                    if last_round_info and "messages" in last_round_info:
                        message = last_round_info["messages"][i]

                    if len(batched_visuals) > 0:
                        visual = batched_visuals[i] if i < len(batched_visuals) else None
                        if isinstance(visual, Image.Image):  # Single image
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            message.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "image": f"data:image/jpeg;base64,{base64_string}",
                                        },
                                        {"type": "text", "text": context},
                                    ],
                                }
                            )
                        elif isinstance(visual, list | tuple) and all(
                            isinstance(v, Image.Image) for v in visual
                        ):  # Multiple images
                            image_content = []
                            for v in visual:
                                base64_image = v.convert("RGB")
                                buffer = BytesIO()
                                base64_image.save(buffer, format="JPEG")
                                base64_bytes = base64.b64encode(buffer.getvalue())
                                base64_string = base64_bytes.decode("utf-8")
                                image_content.append(
                                    {
                                        "type": "image",
                                        "image": f"data:image/jpeg;base64,{base64_string}",
                                    }
                                )
                            message.append(
                                {
                                    "role": "user",
                                    "content": image_content + [{"type": "text", "text": context}],
                                }
                            )
                        else:
                            message.append(
                                {"role": "user", "content": [{"type": "text", "text": context}]}
                            )
                    else:
                        message.append(
                            {"role": "user", "content": [{"type": "text", "text": context}]}
                        )

                    messages.append(message)

                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=True
                    )
                    for msg in messages
                ]
                image_inputs, video_inputs = process_vision_info(messages)

                if video_inputs is not None:
                    raise ValueError(
                        "Video inputs should be empty for current implementation of Qwen2VL."
                    )

                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to("cuda") if self.device_map == "auto" else inputs.to(self.device)
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 128
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                pad_token_id = self.tokenizer.pad_token_id

                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=gen_kwargs["temperature"] > 0,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self._use_cache,
                )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, cont, strict=True)
                ]
                answers = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for i, answer in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            answer = answer.split(term)[0]
                    answers[i] = answer
                    messages[i].append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        }
                    )

                round_idx += 1
                batched_round_results.append(answers)
                batched_round_info.append([dict(messages=messages)])

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("qwen2-vl-7b")
def qwen2_vl_7b(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2-vl-2b")
def qwen2_vl_2b(**model_kwargs) -> Model:
    """Load the Qwen2VL model with 2B params."""
    model_name_or_path = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-7b")
def qwen25_vl_7b(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 7B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model


@register_model("qwen2.5-vl-3b")
def qwen25_vl_3b(**model_kwargs) -> Model:
    """Load the Qwen2.5VL model with 3B params."""
    model_name_or_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2VL(model_name_or_path, **model_kwargs)
    return model
