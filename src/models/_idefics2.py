import os

import torch
from transformers import AutoProcessor, Idefics2ForConditionalGeneration

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

log = utils.get_logger(__name__, rank_zero_only=True)

DEFAULT_IMAGE_TOKEN = "<image>"  # noqa: S105 # nosec B105


class Idefics2(Model):
    """Idefics2 model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "HuggingFaceM4/idefics2-8b".
        attn_implementation (str, optional): The attention implementation to use. Defaults to
            "flash_attention_2" if `flash_attn` is installed, "eager" otherwise.
        do_image_splitting (bool): Whether to split the images in parts. Defaults to False.
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    References:
    ----------
        - https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py

    """

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceM4/idefics2-8b",
        attn_implementation: str | None = "eager",
        do_image_splitting: bool = False,
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._attn_implementation = attn_implementation
        self._do_image_splitting = do_image_splitting

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
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
            "attn_implementation": self._attn_implementation,
        }
        processor_kwargs = {
            "do_image_splitting": self._do_image_splitting,
            "trust_remote_code": os.getenv("HF_TRUST_REMOTE_CODE", False),
        }

        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        self._model = Idefics2ForConditionalGeneration.from_pretrained(
            self._model_name_or_path, **model_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path, **processor_kwargs
        )
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
            (
                contexts,
                all_gen_kwargs,
                doc_to_visuals,
                doc_id,
                tasks,
                splits,
            ) = zip(*chunk, strict=True)

            visuals = [
                doc_to_visual(self.task_dict[task][split][ids])
                for ids, task, split, doc_to_visual in zip(
                    doc_id, tasks, splits, doc_to_visuals, strict=True
                )
            ]

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            gen_kwargs.pop("until", None)
            gen_kwargs.pop("image_aspect_ratio", None)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            prompts = []
            for context, visual in zip(contexts, visuals, strict=True):
                content = []
                if DEFAULT_IMAGE_TOKEN not in context:
                    for _ in visual:
                        content.append({"type": "image"})
                content.append({"type": "text", "text": context})
                message = [{"role": "user", "content": content}]
                prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
                prompts.append(prompt)

            inputs = self.processor(
                text=prompts, images=visuals, padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output_ids = self.model.generate(**inputs, **gen_kwargs)

            # Only retain the generated text
            generated_texts = []
            for output_id, input_id in zip(output_ids, inputs["input_ids"], strict=True):
                generated_id = output_id[len(input_id) :]
                generated_text = self.tokenizer.decode(generated_id, skip_special_tokens=True)
                generated_texts.append(generated_text)

                res.append(generated_text)
            pbar.update(1)

            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), generated_texts)

        # Reorder this group of results back to original unsorted form
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
                doc_to_visuals,
                doc_to_text,
                doc_id,
                tasks,
                splits,
            ) = zip(*chunk, strict=True)
            task, split = tasks[0], splits[0]

            visuals = [
                doc_to_visual(self.task_dict[task][split][ids])
                for ids, task, split, doc_to_visual in zip(
                    doc_id, tasks, splits, doc_to_visuals, strict=True
                )
            ]

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            gen_kwargs.pop("until", None)
            gen_kwargs.pop("image_aspect_ratio", None)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            round_idx = 0
            batched_round_res = []
            batched_previous_round_info = None
            while True:
                # Get current round visual and context from doc_to_text function
                if round_idx != 0:
                    results = []
                    for ids_idx, ids in enumerate(doc_id):
                        prev_outputs = [round_res[ids_idx] for round_res in batched_round_res]

                        prev_info = (
                            None
                            if batched_previous_round_info is None
                            else batched_previous_round_info[ids_idx]
                        )
                        result = doc_to_text[0](
                            self.task_dict[task][split][ids],
                            round_idx=round_idx,
                            prev_outputs=prev_outputs,
                            prev_round_info=prev_info,
                        )
                        results.append(result)

                    (
                        visuals,
                        contexts,
                        batched_terminal_signal,
                        batched_round_res,
                        batched_previous_round_info,
                    ) = list(zip(*results, strict=True))

                    batched_round_res = list(zip(*batched_round_res, strict=True))
                    if batched_terminal_signal[0]:  # Terminal signal from doc_to_text function
                        break

                if isinstance(visuals, tuple):
                    visuals = list(visuals)

                prompts = []
                for context, visual in zip(contexts, visuals, strict=True):
                    content = []
                    if DEFAULT_IMAGE_TOKEN not in context:
                        for _ in visual:
                            content.append({"type": "image"})
                    content.append({"type": "text", "text": context})
                    message = [{"role": "user", "content": content}]
                    prompt = self.processor.apply_chat_template(
                        message, add_generation_prompt=True
                    )
                    prompts.append(prompt)

                inputs = self.processor(
                    text=prompts, images=visuals, padding=True, return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output_ids = self.model.generate(**inputs, **gen_kwargs)

                # Only retain the generated text
                text_outputs = []
                for output_id, input_id in zip(output_ids, inputs["input_ids"], strict=True):
                    generated_id = output_id[len(input_id) :]
                    text_output = self.tokenizer.decode(generated_id, skip_special_tokens=True)
                    text_outputs.append(text_output)

                batched_round_res.append(text_outputs)

                round_idx += 1

            res.extend(list(zip(*batched_round_res, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_res
            )
            pbar.update(1)

        # Reorder this group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("idefics2-8b")
def idefics2_8b(**model_kwargs) -> Model:
    """Load the IDEFICS2 model with 8B params."""
    model_name_or_path = "HuggingFaceM4/idefics2-8b"
    model = Idefics2(model_name_or_path, **model_kwargs)
    return model
