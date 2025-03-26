import os
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    LlavaPreTrainedModel,
)

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

__all__ = ["LLaVA"]

log = utils.get_logger(__name__, rank_zero_only=True)

DEFAULT_IMAGE_TOKEN = "<image>"  # noqa: S105 # nosec B105
VICUNA_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.index0 == 0 %}A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {{ message['content'] }} {% elif message['role'] == 'user' %}USER: {{ message['content'] }} {% else %} ASSISTANT: {{ message['content'] }}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"  # noqa: E501


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


class LLaVA(Model):
    """LLaVA model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models.  Defaults to "llava-hf/llava-1.5-7b-hf".
        attn_implementation (str, optional): The attention implementation to use. Defaults to None.
        chat_template (str, optional): Template for formatting chat conversations. Defaults to
            None.
        use_cache (bool): Whether to use KV cache during generation. Defaults to True.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava

    """

    def __init__(
        self,
        model_name_or_path: str = "llava-hf/llava-1.5-7b-hf",
        attn_implementation: str | None = None,
        chat_template: str | None = None,
        use_cache: bool = True,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._attn_implementation = attn_implementation
        self._chat_template = chat_template
        self._use_cache = use_cache

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
        model_map: dict[str, LlavaPreTrainedModel] = {
            "llava": LlavaForConditionalGeneration,
            "llava_next": LlavaNextForConditionalGeneration,
        }

        try:
            from transformers import LlavaOnevisionForConditionalGeneration

            model_map["llava_onevision"] = LlavaOnevisionForConditionalGeneration
        except ImportError:
            log.warning("Transformers version does not support llava-onevision. Skipping.")

        config = AutoConfig.from_pretrained(self._model_name_or_path)
        model_type = getattr(config, "model_type", "llava")
        model_type = model_map[model_type]

        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "attn_implementation": self._attn_implementation,
        }
        processor_kwargs = {
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
        }

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        self._model = model_type.from_pretrained(self._model_name_or_path, **model_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )

        # Pad from left for batched generation:
        # https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/llava#usage-tips
        self._processor.tokenizer.padding_side = "left"
        self._tokenizer = self._processor.tokenizer

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
        res = []
        pbar_kwargs = dict(total=len(requests), disable=self.rank != 0, desc="Model Responding")
        pbar = utils.get_progress_bar(**pbar_kwargs)

        reg_args = [reg.args for reg in requests]
        for context, doc_to_target, doc_to_visual, doc_id, task, split in reg_args:
            # Encode, pad, and truncate contexts for this batch
            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = _flatten_list(visuals)

            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
            image_tokens = " ".join(image_tokens)
            context = f"{image_tokens}\n{context}"

            # Apply chat template
            messages = [
                {"role": "user", "content": context},
                {"role": "assistant", "content": continuation},
            ]
            if self._chat_template is not None:
                self.tokenizer.chat_template = self._chat_template
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                prompt_and_continuation = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                prompt = self.tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                prompt_and_continuation = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )

            context_fmt = [prompt]
            continuation_fmt = [prompt_and_continuation]
            model_inputs = self.processor(
                text=continuation_fmt, images=visuals, return_tensors="pt"
            ).to(self.device, self.model.dtype)
            labels = model_inputs["input_ids"].clone()
            context_id = self.processor(text=context_fmt, return_tensors="pt")["input_ids"]
            labels[:, : context_id.shape[1]] = -100

            if self.accelerator.is_main_process and doc_id % 100 == 0:
                log.debug("Prompt for doc ID %s:\n\n%s\n", doc_id, context_fmt[0])
                log.debug(
                    "Prompt and continuation for doc ID %s:\n\n%s\n", doc_id, continuation_fmt[0]
                )

            with torch.inference_mode():
                outputs = self.model(**model_inputs, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            continuation_tokens = model_inputs["input_ids"][:, context_id.shape[1] :]
            greedy_tokens = greedy_tokens[
                :, context_id.shape[1] : model_inputs["input_ids"].shape[1]
            ]
            max_equal = (greedy_tokens == continuation_tokens).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)

        pbar.close()
        return res

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
            task, split = task[0], split[0]
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

            # Some benchmarks like MME do not contain image tokens, so we prepend them to the
            # prompt.
            if DEFAULT_IMAGE_TOKEN not in context:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                context = f"{image_tokens}\n{context}"

            # Apply chat template
            messages = [{"role": "user", "content": context}]
            if self._chat_template is not None:
                self.tokenizer.chat_template = self._chat_template
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif self.tokenizer.chat_template is not None:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                log.debug("Prompt for doc ID %s:\n\n%s\n", doc_id[0], text)

            inputs = self.processor(images=visuals, text=text, return_tensors="pt")
            inputs = inputs.to(self.device, self.model.dtype)

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
                use_cache=self._use_cache,
                pad_token_id=self.eot_token_id,
                eos_token_id=self.eot_token_id,
            )
            cont = cont[:, inputs["input_ids"].shape[-1] :]

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]

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
            task, split = task[0], split[0]
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

                if isinstance(visuals, tuple):
                    visuals = list(*visuals)

                # Some benchmarks like MME do not contain image tokens, so we prepend them to the
                # prompt.
                if DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                    image_tokens = " ".join(image_tokens)
                    context = f"{image_tokens}\n{context}"

                # Apply chat template
                messages = [{"role": "user", "content": context}]
                if self._chat_template is not None:
                    self.tokenizer.chat_template = self._chat_template
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                elif self.tokenizer.chat_template is not None:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                    log.debug("Prompt for doc ID %s:\n\n%s\n", doc_id[0], text)

                inputs = self.processor(images=visuals, text=text, return_tensors="pt")
                inputs = inputs.to(self.device, self.model.dtype)

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
                    use_cache=self._use_cache,
                    pad_token_id=self.eot_token_id,
                    eos_token_id=self.eot_token_id,
                )
                cont = cont[:, inputs["input_ids"].shape[-1] :]

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

                round_idx += 1
                batched_round_results.append(text_outputs)

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        # Reorder the group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("llava-next-mistral-7b")
def llava_next_mistral_7b(**model_kwargs) -> Model:
    """Load the LLaVA NeXT model with Mistral 7B params."""
    model_name_or_path = "llava-hf/llava-v1.6-mistral-7b-hf"
    model = LLaVA(model_name_or_path, **model_kwargs)
    return model


@register_model("llava-next-vicuna-7b")
def llava_next_vicuna_7b(**model_kwargs) -> Model:
    """Load the LLaVA NeXT model with Vicuna 7B params."""
    model_name_or_path = "llava-hf/llava-v1.6-vicuna-7b-hf"
    model = LLaVA(model_name_or_path, **model_kwargs)
    return model


@register_model("llava-1.5-13b")
def llava_15_13b(**model_kwargs) -> Model:
    """Load the LLaVA model 13B params."""
    model_name_or_path = "llava-hf/llava-1.5-13b-hf"
    model = LLaVA(model_name_or_path, **model_kwargs)
    return model


@register_model("llava-1.5-7b")
def llava_15_7b(**model_kwargs) -> Model:
    """Load the LLaVA model 7B params."""
    model_name_or_path = "llava-hf/llava-1.5-7b-hf"
    model = LLaVA(model_name_or_path, **model_kwargs)
    return model
