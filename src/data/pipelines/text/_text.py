import torch

__all__ = [
    "concept_extraction_spacy",
    "encode_sentence_bert",
    "elo_score_llama32",
    "textual_inclusion_llama32",
]

llama_32_model = None

sentence_bert_model = None
sentence_bert_processor = None

spacy_model = None


def concept_extraction_spacy(batch: dict, **kwargs) -> dict:
    """Extract concepts from text using spaCy.

    Args:
    ----
        batch (dict): The batch of data containing text to process.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: Additional keyword arguments.

    """
    import spacy

    input_column = kwargs.pop("input_column", "text")
    output_column = kwargs.pop("output_column", f"{input_column}_concepts")
    skip_words = kwargs.pop("skip_words", [])
    remove_prefix_words = kwargs.pop("remove_prefix_words", False)

    global spacy_model
    if spacy_model is None:
        import spacy

        try:
            spacy_model = spacy.load("en_core_web_lg")
        except OSError:
            import spacy.cli

            spacy.cli.download("en_core_web_lg")
            spacy_model = spacy.load("en_core_web_lg")

    if input_column not in batch:
        raise ValueError(f"{input_column} missing in dataset")

    if isinstance(batch[input_column], list):
        text = batch[input_column]

        docs = spacy_model.pipe(text, batch_size=len(text))

        all_concepts = []
        for doc in docs:
            concepts = []

            for chunk in doc.noun_chunks:
                concept = chunk.text.lower()

                if remove_prefix_words:
                    articles = ["a", "an", "the"]
                    possessive_pronouns = ["his", "her", "its", "their"]

                    for term in [*articles, *possessive_pronouns]:
                        if concept.startswith(term + " "):
                            concept = concept[len(term) + 1 :]
                            break

                    if concept in skip_words:
                        continue

                    concepts.append(concept)

            for ent in doc.ents:
                concept = ent.text.lower()

                if remove_prefix_words:
                    articles = ["a", "an", "the"]
                    possessive_pronouns = ["his", "her", "its", "their"]

                    for term in [*articles, *possessive_pronouns]:
                        if concept.startswith(term + " "):
                            concept = concept[len(term) + 1 :]
                            break

                    if concept in skip_words:
                        continue

                if concept not in concepts:
                    concepts.append(concept)

            all_concepts.append(concepts)

        batch[output_column] = all_concepts
    else:
        text = batch[input_column]

        doc = spacy_model(text)
        concepts = []

        for chunk in doc.noun_chunks:
            concept = chunk.text.lower()

            if remove_prefix_words:
                articles = ["a", "an", "the"]
                possessive_pronouns = ["his", "her", "its", "their"]

                for term in [*articles, *possessive_pronouns]:
                    if concept.startswith(term + " "):
                        concept = concept[len(term) + 1 :]
                        break

                if concept in skip_words:
                    continue

                concepts.append(concept)

        for ent in doc.ents:
            concept = ent.text.lower()

            if remove_prefix_words:
                articles = ["a", "an", "the"]
                possessive_pronouns = ["his", "her", "its", "their"]

                for term in [*articles, *possessive_pronouns]:
                    if concept.startswith(term + " "):
                        concept = concept[len(term) + 1 :]
                        break

                if concept in skip_words:
                    continue

            if concept not in concepts:
                concepts.append(concept)

        batch[output_column] = concepts

    return batch


def encode_sentence_bert(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Encode text with the Sentence-BERT model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    input_column = kwargs.pop("input_column", "text")
    output_column = kwargs.pop("output_column", f"{input_column}_sentence_bert_embeds")

    global sentence_bert_model
    global sentence_bert_processor
    if sentence_bert_model is None:
        from transformers import AutoModel, AutoTokenizer

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        sentence_bert_processor = AutoTokenizer.from_pretrained(model_name)
        sentence_bert_model = AutoModel.from_pretrained(model_name)

    if rank is not None or torch.cuda.is_available():
        device = f"cuda:{(rank or 0)% torch.cuda.device_count()}"
        dtype = torch.float16 if device != "cpu" else torch.float32
        sentence_bert_model.to(device=device, dtype=dtype)
    else:
        device = "cpu"

    if input_column not in batch:
        raise ValueError(f"{input_column} missing in dataset")

    def _mean_pooling(text_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling of the token embeddings.

        Args:
        ----
            text_embeds (torch.Tensor): The token embeddings.
            attention_mask (torch.Tensor): The attention mask.

        """
        # The first element of model_output contains all token embeddings
        token_embeddings = text_embeds[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    if isinstance(batch[input_column], list):
        text = batch[input_column]
        text_inputs = sentence_bert_processor(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        text_inputs = text_inputs.to(device=sentence_bert_model.device)
        with torch.no_grad():
            text_embeds = sentence_bert_model(**text_inputs)
            text_embeds = _mean_pooling(text_embeds, text_inputs["attention_mask"])

        # Normalize the embeddings
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        batch[output_column] = text_embeds.cpu().numpy().tolist()
    else:
        raise NotImplementedError

    return batch


def _score_pair_llama32(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Give a score to (reference, prediction) pairs with the Llama 3.2 model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    reference_column = kwargs.pop("reference_column", "reference")
    prediction_column = kwargs.pop("prediction_column", "prediction")
    output_column = kwargs.pop("output_column", "pair_score")
    question_template = kwargs.pop("question_template")

    global llama_32_model
    if llama_32_model is None:
        from transformers import pipeline

        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        llama_32_model = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        llama_32_model.generation_config.pad_token_id = llama_32_model.model.config.eos_token_id[0]

    if reference_column not in batch:
        raise ValueError(f"{reference_column} missing in dataset")
    if prediction_column not in batch:
        raise ValueError(f"{prediction_column} missing in dataset")
    if not isinstance(question_template, str):
        raise ValueError("`question_template` must be defined!")

    if isinstance(batch[reference_column], list):
        references = batch[reference_column]
        predictions = batch[prediction_column]

        inputs = [
            [{"role": "user", "content": question_template % (pred, ref)}]
            for pred, ref in zip(predictions, references, strict=True)
        ]
        outputs = llama_32_model(
            inputs,
            max_new_tokens=16,
            do_sample=False,
            top_p=None,
            temperature=None,
        )
        outputs = [output[0]["generated_text"][-1]["content"] for output in outputs]

        batch[output_column] = outputs

    else:
        raise NotImplementedError

    return batch


def _score_triplet_llama32(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Give a score to (reference, prediction_a, prediction_b) triplets with the Llama 3.2 model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    reference_column = kwargs.pop("reference_column", "reference")
    prediction_a_column = kwargs.pop("prediction_a_column", "prediction_a")
    prediction_b_column = kwargs.pop("prediction_b_column", "prediction_b")
    output_column = kwargs.pop("output_column", "triplet_score")
    question_template = kwargs.pop("question_template")

    global llama_32_model
    if llama_32_model is None:
        from transformers import pipeline

        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        llama_32_model = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        llama_32_model.generation_config.pad_token_id = llama_32_model.model.config.eos_token_id[0]

    if reference_column not in batch:
        raise ValueError(f"{reference_column} missing in dataset")
    if prediction_a_column not in batch:
        raise ValueError(f"{prediction_a_column} missing in dataset")
    if prediction_b_column not in batch:
        raise ValueError(f"{prediction_b_column} missing in dataset")
    if not isinstance(question_template, str):
        raise ValueError("`question_template` must be defined!")

    if isinstance(batch[reference_column], list):
        references = batch[reference_column]
        predictions_a = batch[prediction_a_column]
        predictions_b = batch[prediction_b_column]

        inputs = [
            [{"role": "user", "content": question_template % (pred_a, pred_b, ref)}]
            for pred_a, pred_b, ref in zip(predictions_a, predictions_b, references, strict=True)
        ]
        outputs = llama_32_model(
            inputs,
            max_new_tokens=16,
            do_sample=False,
            top_p=None,
            temperature=None,
        )
        outputs = [output[0]["generated_text"][-1]["content"] for output in outputs]

        batch[output_column] = outputs

    else:
        raise NotImplementedError

    return batch


def elo_score_llama32(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Give an score to (reference, prediction_a, prediction_b) triplets with the Llama 3.2 model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    if not kwargs.get("question_template"):
        question_template = (
            "You are a model that discriminates whether labels A or B better align with a target"
            " value.\n"
            "\n"
            "This is label A: %s\n"
            "This is label B: %s\n"
            "This is the target value: %s\n"
            "\n"
            "Does A align better with the target value? Does B align better with the target value?"
            ' Reply only with "1" if A wins over B, or "0" if B wins over A.'
        )
        kwargs["question_template"] = question_template

    if not kwargs.get("output_column"):
        kwargs["output_column"] = "elo_score"

    return _score_triplet_llama32(batch, rank=rank, **kwargs)


def textual_inclusion_llama32(batch: dict, rank: int | None = None, **kwargs) -> dict:
    """Give an inclusion score to (reference, prediction) pairs with the Llama 3.2 model.

    Args:
    ----
        batch (dict): The batch of data.
        rank (int, optional): The rank of the process. Defaults to None.
        **kwargs: The keyword arguments.

    """
    if not kwargs.get("question_template"):
        question_template = (
            "You are a model that determines whether an answer is a good reply to a question"
            " given also its target value.\n"
            "\n"
            "This is the question: What type of object is in this photo?\n"
            "This is the answer: %s\n"
            "This is the target value: %s\n"
            "\n"
            "If the answer describes the target, reply positively."
            " If the answer includes the target value or a synonym of it, reply positively."
            " If the target is generic but it is related to the answer, reply positively."
            ' Reply only with "1" if yes, or "0" if no.'
        )
        kwargs["question_template"] = question_template

    if not kwargs.get("output_column"):
        kwargs["output_column"] = "exact_match_score"

    return _score_pair_llama32(batch, rank=rank, **kwargs)
