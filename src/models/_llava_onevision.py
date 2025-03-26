import copy
import json

import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from packaging import version

from src import utils
from src.data.tasks import TaskInstance
from src.models._api import register_model
from src.models._base import Model

__all__ = ["LLaVAOnevision"]

log = utils.get_logger(__name__, rank_zero_only=True)


def _is_json(string: str) -> bool:
    """Check if a string is a valid JSON.

    Args:
    ----
        string (str): The string to check.

    """
    try:
        json.loads(string)
        return True
    except json.JSONDecodeError:
        return False


class LLaVAOnevision(Model):
    """LLaVA-Onevision model.

    Args:
    ----
        model_name_or_path (str): Path to pretrained model or model identifier from
            huggingface.co/models. Defaults to "lmms-lab/llava-onevision-qwen2-7b-ov".
        model_name (str, optional): Name of the model. Defaults to None.
        attn_implementation (str, optional): The attention implementation to use. Defaults to
            "sdpa" for torch>=2.1.2, "eager" otherwise.
        conv_template (str, optional): Template for formatting conversations. Defaults to
            "qwen_1_5".
        use_flash_attn_2 (bool): Whether to use flash attention 2. Defaults to False.
        use_cache (bool, optional): Whether to use KV cache during generation. Defaults to True.
        customized_config (str, optional): Path to custom configuration JSON file. Defaults to
            None.
        mm_spatial_pool_stride (int, optional): Stride for spatial pooling. Defaults to 2.
        mm_spatial_pool_mode (str, optional): Mode for spatial pooling. Defaults to "bilinear".
        batch_size (int): Batch size for model inference. Defaults to 1.
        device_map (str): Device map for model parallel loading. Defaults to "auto".
        dtype (str | torch.dtype): Data type for model weights. Defaults to "torch.bfloat16".
        load_in_8bit (bool, optional): Whether to load the model in 8-bit. Defaults to False.
        load_in_4bit (bool, optional): Whether to load the model in 4-bit. Defaults to False.
        kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        model_name_or_path: str = "lmms-lab/llava-onevision-qwen2-7b-ov",
        model_name: str | None = None,
        attn_implementation: str | None = (
            "sdpa" if version.parse(torch.__version__) >= version.parse("2.1.2") else "eager"
        ),
        use_flash_attn_2: bool = False,
        conv_template: str | None = "qwen_1_5",
        use_cache: bool | None = True,
        customized_config: str | None = None,
        mm_spatial_pool_stride: int | None = 2,
        mm_spatial_pool_mode: str | None = "bilinear",
        batch_size: int = 1,
        device_map: str = "auto",
        dtype: str | torch.dtype = "bfloat16",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> None:
        self._model_name_or_path = model_name_or_path
        self._model_name = (
            model_name if model_name is not None else get_model_name_from_path(model_name_or_path)
        )
        self._attn_implementation = attn_implementation
        self._use_flash_attn_2 = use_flash_attn_2
        self._conv_template = conv_template
        self._use_cache = use_cache
        self._customized_config = customized_config
        self._mm_spatial_pool_stride = mm_spatial_pool_stride
        self._mm_spatial_pool_mode = mm_spatial_pool_mode

        self._max_length = None

        super().__init__(
            batch_size=batch_size,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            distributed_types=["FSDP", "MULTI_GPU", "DEEPSPEED"],
            **kwargs,
        )

        torch.backends.cuda.matmul.allow_tf32 = True

    def _pad_sequence(
        self, input_ids: list[torch.Tensor], batch_first: bool, padding_value: int
    ) -> torch.Tensor:
        """Pad a list of variable length tensors.

        Custom padder that handles left and right padding based on tokenizer setting.
        Left-padding tensors are flipped before and after the padding operation.

        Args:
        ----
            input_ids (list[torch.Tensor]): List of input tensors to pad.
            batch_first (bool): Whether output should be batch first (B, T) or sequence first
                (T, B).
            padding_value (int): Value used for padding.

        """
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

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
        try:
            return self.tokenizer.decode(tokens)
        except ValueError:
            return self.tokenizer.decode([tokens])

    def load_model(self) -> None:
        """Load the model in memory."""
        model_kwargs = {
            "multimodal": True,
            "use_flash_attention_2": self._use_flash_attn_2,
            "torch_dtype": str(self.dtype).split(".")[1],
            "device_map": self.device_map,
        }

        if self._customized_config is not None:
            model_kwargs["customized_config"] = self._customized_config
        if self._attn_implementation is not None:
            model_kwargs["attn_implementation"] = self._attn_implementation
        if self._quantization_config is not None:
            model_kwargs["quantization_config"] = self._quantization_config

        overwrite_config = {}
        overwrite_config["mm_spatial_pool_stride"] = self._mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_mode"] = self._mm_spatial_pool_mode

        model_kwargs["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multi-modal argument
            tokenizer, model, processor, max_length = load_pretrained_model(
                self._model_name_or_path,
                None,
                self._model_name,
                **model_kwargs,
            )
            self._model = model
            self._processor = processor
            self._tokenizer = tokenizer
            self._max_length = max_length
        except TypeError:
            # Older versions of LLaVA don't have multi-modal argument
            model_kwargs.pop("multimodal", None)
            tokenizer, model, processor, max_length = load_pretrained_model(
                self._model_name_or_path,
                None,
                self._model_name,
                **model_kwargs,
            )
            self._model = model
            self._processor = processor
            self._tokenizer = tokenizer
            self._max_length = max_length

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
        origin_image_aspect_ratio = getattr(self.config, "image_aspect_ratio", None)

        reg_args = [reg.args for reg in requests]
        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in reg_args:
            visual = doc_to_visual(self.task_dict[task][split][doc_id])

            wrong_aspect_ratio = self.config.image_aspect_ratio != origin_image_aspect_ratio
            if origin_image_aspect_ratio is not None and wrong_aspect_ratio:
                self.config.image_aspect_ratio = origin_image_aspect_ratio
                log.info("Resetting image aspect ratio to %s", origin_image_aspect_ratio)

            if visual is None or visual == []:
                visual = None
                image_tensor = None
            else:
                if len(visual) > 1 or "image_aspect_ratio" not in self.config.__dict__:
                    self.config.image_aspect_ratio = "pad"
                    log.info(
                        "In Multi-Image setting, image aspect ratio: %s",
                        self.config.image_aspect_ratio,
                    )

                image_tensor = process_images(visual, self.processor, self.config)
                if isinstance(image_tensor, list):
                    image_tensor = [
                        _image.to(dtype=self.dtype, device=self.device) for _image in image_tensor
                    ]
                else:
                    image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

            is_image_defined = image_tensor is not None and len(image_tensor) != 0
            if is_image_defined and DEFAULT_IMAGE_TOKEN not in contexts:
                placeholder_count = len(visual) if isinstance(visual, list) else 1
                image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + contexts
            else:
                prompts_input = contexts

            if "llama_3" in self._conv_template:
                conv = copy.deepcopy(conv_templates[self._conv_template])
            else:
                conv = conv_templates[self._conv_template].copy()

            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            if isinstance(doc_to_target, str):
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])

            conv.messages[-1][1] = continuation
            full_prompt = conv.get_prompt()
            full_input_ids = (
                tokenizer_image_token(
                    full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .to(self.device)
            )

            labels = full_input_ids.clone()
            labels[0, : input_ids.shape[1]] = -100

            if visual is None:
                raise ValueError("visual cannot be None.")

            kwargs = {}
            kwargs["image_sizes"] = (
                [[v.size[0], v.size[1]] for v in visual]
                if isinstance(visual, list)
                else [[visual.size[0], visual.size[1]]]
            )

            with torch.inference_mode():
                outputs = self.model(
                    input_ids=full_input_ids,
                    labels=labels,
                    images=image_tensor,
                    use_cache=True,
                    **kwargs,
                )

            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            continuation_tokens = full_input_ids[:, input_ids.shape[1] :]
            greedy_tokens = greedy_tokens[:, input_ids.shape[1] : full_input_ids.shape[1]]
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

        origin_image_aspect_ratio = getattr(self.config, "image_aspect_ratio", None)

        pbar_kwargs = dict(total=num_iters, disable=self.rank != 0, desc="Model Responding")
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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []
            for visual, context in zip(batched_visuals, batched_contexts, strict=True):
                wrong_aspect_ratio = self.config.image_aspect_ratio != origin_image_aspect_ratio
                if origin_image_aspect_ratio is not None and wrong_aspect_ratio:
                    self.config.image_aspect_ratio = origin_image_aspect_ratio
                    log.info("Resetting image aspect ratio to %s", origin_image_aspect_ratio)

                if visual is None or visual == []:  # For text-only tasks.
                    visual = None
                    placeholder_count = 0
                    image_tensor = None
                else:
                    # For multi image case, we treat per image aspect ratio as "pad" by default.
                    if len(visual) > 1 or "image_aspect_ratio" not in self.config.__dict__:
                        self.config.image_aspect_ratio = getattr(
                            gen_kwargs, "image_aspect_ratio", "pad"
                        )
                        log.info(
                            "In Multi-Image setting, image aspect ratio: %s",
                            self.config.image_aspect_ratio,
                        )

                    image_tensor = process_images(visual, self.processor, self.config)
                    if isinstance(image_tensor, list):
                        image_tensor = [
                            _image.to(dtype=self.dtype, device=self.device)
                            for _image in image_tensor
                        ]
                    else:
                        image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

                    placeholder_count = len(visual) if isinstance(visual, list) else 1

                is_image_defined = image_tensor is not None and len(image_tensor) != 0
                if is_image_defined and DEFAULT_IMAGE_TOKEN not in context:
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self._conv_template:
                    conv = copy.deepcopy(conv_templates[self._conv_template])
                else:
                    conv = conv_templates[self._conv_template].copy()

                if _is_json(question):  # Conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    if len(conv.messages) % 2 != 1:
                        raise ValueError("Number of messages must be odd.")

                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # Only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # Pre-configure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                for prompt in question_input
            ]
            pad_token_ids = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            input_ids = self._pad_sequence(
                input_ids_list, batch_first=True, padding_value=pad_token_ids
            ).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            gen_kwargs["image_sizes"] = [
                batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))
            ]

            # These steps are not in LLaVA's original code, but are necessary for generation
            # to work.
            # TODO attention to this major generation step...
            if "image_aspect_ratio" in gen_kwargs:
                gen_kwargs.pop("image_aspect_ratio")

            with torch.inference_mode():
                cont = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    use_cache=self._use_cache,
                    **gen_kwargs,
                )

            text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)

            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)

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

        origin_image_aspect_ratio = getattr(self.config, "image_aspect_ratio", None)

        pbar_kwargs = dict(total=num_iters, disable=self.rank != 0, desc="Model Responding")
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

            # Assume all gen kwargs in the batch are the same
            # This is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

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

                question_input = []
                for visual, context in zip(batched_visuals, batched_contexts, strict=True):
                    wrong_aspect_ratio = (
                        self.config.image_aspect_ratio != origin_image_aspect_ratio
                    )
                    if origin_image_aspect_ratio is not None and wrong_aspect_ratio:
                        self.config.image_aspect_ratio = origin_image_aspect_ratio
                        log.info("Resetting image aspect ratio to %s", origin_image_aspect_ratio)

                    if visual is None or visual == []:  # For text-only tasks.
                        visual = None
                        placeholder_count = 0
                        image_tensor = None
                    else:
                        # For multi image case, we treat per image aspect ratio as "pad" by
                        # default.
                        if len(visual) > 1 or "image_aspect_ratio" not in self.config.__dict__:
                            self.config.image_aspect_ratio = getattr(
                                gen_kwargs, "image_aspect_ratio", "pad"
                            )
                            log.info(
                                "In Multi-Image setting, image aspect ratio: %s",
                                self.config.image_aspect_ratio,
                            )

                        image_tensor = process_images(visual, self.processor, self.config)
                        if isinstance(image_tensor, list):
                            image_tensor = [
                                _image.to(dtype=self.dtype, device=self.device)
                                for _image in image_tensor
                            ]
                        else:
                            image_tensor = image_tensor.to(dtype=self.dtype, device=self.device)

                        placeholder_count = len(visual) if isinstance(visual, list) else 1

                    is_image_defined = image_tensor is not None and len(image_tensor) != 0
                    if is_image_defined and DEFAULT_IMAGE_TOKEN not in context:
                        image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                        image_tokens = " ".join(image_tokens)
                        question = image_tokens + "\n" + context
                    else:
                        question = context

                    # This is much safer for llama3, as we now have some object type in it
                    if "llama_3" in self._conv_template:
                        conv = copy.deepcopy(conv_templates[self._conv_template])
                    else:
                        conv = conv_templates[self._conv_template].copy()

                    if last_round_info and "conv" in last_round_info:
                        conv = last_round_info["conv"][0]

                    if _is_json(question):  # Conversational question input
                        question = json.loads(question)
                        for idx, item in enumerate(question):
                            role = conv.roles[idx % 2]
                            message = item["value"]
                            conv.append_message(role, message)

                        if len(conv.messages) % 2 != 1:
                            raise ValueError("Number of messages must be odd.")

                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)
                    else:  # Only simple string for question
                        conv.append_message(conv.roles[0], question)
                        conv.append_message(conv.roles[1], None)
                        prompt_question = conv.get_prompt()
                        question_input.append(prompt_question)

                # Pre-configure gen_kwargs with defaults
                if "max_new_tokens" not in gen_kwargs:
                    gen_kwargs["max_new_tokens"] = 1024
                if "temperature" not in gen_kwargs:
                    gen_kwargs["temperature"] = 0
                if "do_sample" not in gen_kwargs:
                    gen_kwargs["do_sample"] = False
                if "top_p" not in gen_kwargs:
                    gen_kwargs["top_p"] = None
                if "num_beams" not in gen_kwargs:
                    gen_kwargs["num_beams"] = 1

                input_ids_list = [
                    tokenizer_image_token(
                        prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    )
                    for prompt in question_input
                ]
                pad_token_ids = (
                    self.tokenizer.pad_token_id
                    if self.tokenizer.pad_token_id is not None
                    else self.tokenizer.eos_token_id
                )
                input_ids = self._pad_sequence(
                    input_ids_list, batch_first=True, padding_value=pad_token_ids
                ).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)

                if batched_visuals[0] is not None:
                    gen_kwargs["image_sizes"] = [
                        batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))
                    ]

                # These steps are not in LLaVA's original code, but are necessary for generation
                # to work.
                # TODO attention to this major generation step...
                if "image_aspect_ratio" in gen_kwargs:
                    gen_kwargs.pop("image_aspect_ratio")

                if last_round_info and "image_sizes" in last_round_info:
                    gen_kwargs["image_sizes"] = last_round_info["image_sizes"][0]

                if image_tensor is None and last_round_info and "image_tensor" in last_round_info:
                    image_tensor = last_round_info["image_tensor"][0]

                with torch.inference_mode():
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        use_cache=self._use_cache,
                        **gen_kwargs,
                    )

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                text_outputs = [response.strip() for response in text_outputs]
                conv.messages[-1][1] = text_outputs[0]

                round_idx += 1
                batched_round_results.append(text_outputs)
                batched_round_info.append(
                    [
                        dict(
                            conv=[conv],
                            image_tensor=[image_tensor],
                            image_sizes=[gen_kwargs["image_sizes"]],
                        )
                    ]
                )

            res.extend(list(zip(*batched_round_results, strict=True)))

            self.cache_hook.add_partial(
                "generate_until_multi_round", (context, gen_kwargs), batched_round_results
            )
            pbar.update(1)

        # Reorder this group of results back to original unsorted form
        res = reordered.get_original(res)

        pbar.close()
        return res


@register_model("llava-onevision-qwen2-7b-ov")
def llava_onevision_qwen2_7b_ov(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 7B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
    conv_template = "qwen_1_5"
    model = LLaVAOnevision(model_name_or_path, conv_template=conv_template, **model_kwargs)
    return model


@register_model("llava-onevision-qwen2-7b-si")
def llava_onevision_qwen2_7b_si(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 7B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-7b-si"
    conv_template = "qwen_1_5"
    model = LLaVAOnevision(model_name_or_path, conv_template=conv_template, **model_kwargs)
    return model


@register_model("llava-onevision-qwen2-0.5b-ov")
def llava_onevision_qwen2_0_5b_ov(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 0.5B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    conv_template = "qwen_1_5"
    model_name = "llava_qwen"
    model = LLaVAOnevision(
        model_name_or_path,
        conv_template=conv_template,
        model_name=model_name,
        **model_kwargs,
    )
    return model


@register_model("llava-onevision-qwen2-0.5b-si")
def llava_onevision_qwen2_0_5b_si(**model_kwargs) -> Model:
    """Load the LLaVAOnevision model with Qwen2 0.5B."""
    model_name_or_path = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    conv_template = "qwen_1_5"
    model_name = "llava_qwen"
    model = LLaVAOnevision(
        model_name_or_path,
        conv_template=conv_template,
        model_name=model_name,
        **model_kwargs,
    )
    return model
