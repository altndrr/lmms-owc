import math
from collections.abc import Iterable
from typing import Literal

import datasets
import numpy as np
import torch

from src import utils
from src.data.metrics._api import register_aggregation

__all__ = [
    "GROUP_METRICS",
    "bits_per_byte",
    "bleu",
    "brier_score",
    "bypass",
    "chrf",
    "concept_semantic_similarity",
    "f1_score",
    "matthews_corrcoef",
    "mean",
    "mean_average_semantic_similarity",
    "median",
    "perplexity",
    "semantic_similarity",
    "ter",
    "textual_inclusion_llama32",
    "weighted_perplexity",
]

GROUP_METRICS = [
    "bits_per_byte",
    "bleu",
    "brier_score",
    "bypass",
    "chrf",
    "concept_semantic_similarity",
    "f1_score",
    "matthews_corrcoef",
    "mean_average_semantic_similarity",
    "perplexity",
    "semantic_similarity",
    "ter",
    "textual_inclusion_llama32",
    "weighted_perplexity",
]

log = utils.get_logger(__name__, rank_zero_only=True)


def _weighted_mean(items: list) -> float:
    """Calculate the weighted mean of a list of documents.

    Args:
    ----
        items (list): List of documents.

    """
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)


@register_aggregation("bits_per_byte")
def bits_per_byte(items: list) -> float:
    """Calculate the bits per byte on a list of documents.

    Note: bits per byte must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    return -_weighted_mean(items) / math.log(2)


def _sacreformat(refs: list, preds: list) -> tuple:
    """Format refs and preds for sacrebleu corpus calculation.

    Args:
    ----
        refs (list): List of references.
        preds (list): List of predictions.

    """
    if not isinstance(refs, Iterable) or isinstance(refs, str):
        refs = list(refs)
    if not isinstance(refs[0], Iterable) or isinstance(refs[0], str):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs, strict=True))

    if not isinstance(preds, Iterable) or isinstance(preds, str):
        preds = list(preds)
    if not isinstance(preds[0], Iterable) or isinstance(preds[0], str):
        if len(preds) != 1:
            raise ValueError(f"Pred must be a str, found {preds}")
        preds = [pred[0] for pred in preds]

    return refs, preds


@register_aggregation("bleu")
def bleu(items: list) -> float:
    """Calculate the BLEU score on a list of documents.

    Note: BLEU score must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    import sacrebleu

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_aggregation("brier_score")
def brier_score(items: list) -> float:
    """Calculate the Brier score on a list of documents.

    Note: Brier score must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    gold, predictions = list(zip(*items, strict=True))
    bs, num_class = np.array(predictions).shape

    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))


@register_aggregation("bypass")
def bypass(arr: list) -> int:
    """Skip aggregation and return a constant.

    Args:
    ----
        arr (list): List of numbers.

    """
    return 999


@register_aggregation("chrf")
def chrf(items: list) -> float:
    """Calculate the chrF score on a list of documents.

    Note: chrF score must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    import sacrebleu

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_aggregation("concept_semantic_similarity")
def concept_semantic_similarity(
    items: list, reduce: Literal["none", "max", "mean", "median", "min"] = "max"
) -> float | list[tuple[str, float]]:
    """Calculate the semantic similarity of the response concepts on a list of documents.

    Note: concept semantic similarity can be calculated independently per each document in a
    benchmark. However, we evaluate it as a group-level metric as we can batch samples in the
    forward to the spaCy model and the SentenceBERT model to speed up evaluation. For this reason,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.
        reduce ("none" | "max" | "mean" | "median" | "min"): Reduction operation on the values.
            When "none", returns the values per sample; when "max", selects the max similarity over
            the concepts and evaluates the mean over all samples; when "mean" evaluates the mean
            over the concepts and the mean over all samples; when "median" evaluate the median over
            the concepts and the mean over all samples; when "mean", selects the min similarity
            over the concepts and evaluate the mean over all samples. Defaults to "max".

    """
    from src.data.pipelines.text import concept_extraction_spacy, encode_sentence_bert

    if reduce not in ["none", "max", "mean", "median", "min"]:
        raise ValueError(
            "Unknown `reduce` value for `concept_semantic_similarity` metric."
            ' Expected "none", "max", "mean", "median", or "min", but got "%s"',
            reduce,
        )

    skip_words_groups = {
        # Numerical terms
        "numbers_digits": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "numbers_words": [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ],
        # Basic elements
        "symbols": ["*"],
        "articles": ["a", "the"],
        # Common nouns
        "generic_nouns": ["image", "object", "photo", "type", "this photo"],
        # Pronouns and determiners
        "personal_pronouns": ["it", "they", "them"],
        "demonstratives": ["that", "this", "those"],
        # Question words and relatives
        "wh_words": ["which", "who", "whom", "whose", "where", "when", "what", "why", "how"],
        # Quantifiers
        "quantifiers": ["some"],
    }
    skip_words = [word for category in skip_words_groups.values() for word in category]

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]

    refs = [ref[0] if isinstance(ref, list) else ref for ref in refs]
    preds = [pred[-1] if isinstance(pred, list) else pred for pred in preds]

    data = datasets.Dataset.from_dict({"prediction": preds, "reference": refs})
    data.set_format("torch")

    # Extract concepts from the predictions
    data = data.map(
        concept_extraction_spacy,
        batched=True,
        batch_size=1024,
        fn_kwargs={
            "input_column": "prediction",
            "output_column": "prediction_concepts",
            "skip_words": skip_words,
            "remove_prefix_words": True,
        },
    )

    # Add the entire prediction as an extra concept
    data = data.map(
        lambda x: {"prediction_concepts": x["prediction_concepts"] + [x["prediction"]]}
    )

    # Get the reference-concept pairs (also include the (ref, pred) pair).
    ref_concept_pairs = [
        [(ref, concept) for concept in concepts]
        for ref, concepts in zip(data["reference"], data["prediction_concepts"], strict=True)
    ]

    # Get the unique pairs and associate pairs to an index
    unique_ref_concept_pairs = set(sum(ref_concept_pairs, []))
    unique_ref_concept_pairs_to_idx = {
        " | ".join(pair): idx for idx, pair in enumerate(unique_ref_concept_pairs)
    }

    # Make a dataset of unique references and concepts
    pairs_data = datasets.Dataset.from_dict(
        {
            "_pair_idx": range(len(unique_ref_concept_pairs)),
            "reference": [pair[0] for pair in unique_ref_concept_pairs],
            "concept": [pair[1] for pair in unique_ref_concept_pairs],
        }
    )
    pairs_data.set_format("torch")

    # Encode pairs with sentence bert
    pairs_data = pairs_data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "reference"},
    )
    pairs_data = pairs_data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "concept"},
    )

    # Get the semantic similarities
    refs_z = pairs_data["reference_sentence_bert_embeds"].unsqueeze(1)
    concepts_z = pairs_data["concept_sentence_bert_embeds"].unsqueeze(2)
    similarities = torch.bmm(refs_z, concepts_z).squeeze()

    # Get the similarities for each of the unique pairs.
    ref_concept_pairs_idxs = [
        [unique_ref_concept_pairs_to_idx[" | ".join([ref, concept])] for concept in concepts]
        for ref, concepts in zip(data["reference"], data["prediction_concepts"], strict=True)
    ]
    data = data.add_column("ref_concept_pairs_idxs", ref_concept_pairs_idxs)
    data = data.map(
        lambda x: {"concepts_similarities": similarities[np.array(x["ref_concept_pairs_idxs"])]},
        remove_columns=["ref_concept_pairs_idxs"],
    )

    if reduce == "max":
        data = data.map(lambda x: {"max_concept_similarity": x["concepts_similarities"].max()})
        return torch.mean(data["max_concept_similarity"]).item()
    elif reduce == "mean":
        data = data.map(lambda x: {"mean_concept_similarity": x["concepts_similarities"].mean()})
        return torch.mean(data["mean_concept_similarity"]).item()
    elif reduce == "median":
        data = data.map(
            lambda x: {"median_concept_similarity": x["concepts_similarities"].median()}
        )
        return torch.mean(data["median_concept_similarity"]).item()
    elif reduce == "min":
        data = data.map(lambda x: {"min_concept_similarity": x["concepts_similarities"].min()})
        return torch.mean(data["min_concept_similarity"]).item()

    # Return without reduction
    concepts = data["prediction_concepts"]
    similarities = [row.tolist() for row in data["concepts_similarities"]]
    return list(zip(concepts, similarities, strict=True))


@register_aggregation("f1")
def f1_score(items: list) -> float:
    """Calculate the F1 score on a list of documents.

    Note: F1 score must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items, strict=True))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)

    return np.max(fscore)


@register_aggregation("matthews_corrcoef")
def matthews_corrcoef(items: list) -> float:
    """Calculate the Matthews correlation coefficient on a list of documents.

    Note: Matthews correlation coefficient must be calculated across all documents in a benchmark.
    For simplicity, we define it as an aggregation metric and pair it with a no-op passthrough
    metric functions.

    Args:
    ----
        items (list): List of documents.

    """
    from sklearn.metrics import matthews_corrcoef

    unzipped_list = list(zip(*items, strict=True))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


@register_aggregation("mean")
def mean(arr: list) -> float:
    """Calculate the mean of a list of numbers.

    Args:
    ----
        arr (list): List of numbers.

    """
    return sum(arr) / len(arr)


@register_aggregation("mean_average_semantic_similarity")
def mean_average_semantic_similarity(
    items: list, reduce: Literal["none", "mean"] = "mean"
) -> dict[str, float] | dict[str, list[float]]:
    """Calculate the mean average semantic similarity on a list of documents.

    Note: mean average semantic similarity can be calculated independently per each document in a
    benchmark. However, we evaluate it as a group-level metric as we can batch samples in the
    forward to the SentenceBERT model to speed up evaluation. For this reason, we define it as an
    aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.
        reduce ("none" | "mean"): Reduction operation on the values. When "none", returns the
            value per sample; when "mean", returns the mean similarity. Defaults to "mean".

    """
    from src.data.pipelines.text import encode_sentence_bert

    if reduce not in ["none", "mean"]:
        raise ValueError(
            "Unknown `reduce` value for `mean_average_semantic_similarity` metric."
            ' Expected "none" or "mean", but got "%s"',
            reduce,
        )

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]

    refs = [ref[0] if isinstance(ref, list) else ref for ref in refs]
    preds = [pred[-1] if isinstance(pred, list) else pred for pred in preds]

    data = datasets.Dataset.from_dict({"prediction": preds, "reference": refs})
    data.set_format("torch")

    data = data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "reference"},
    )
    data = data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "prediction"},
    )

    refs_z = data["reference_sentence_bert_embeds"].unsqueeze(1)
    preds_z = data["prediction_sentence_bert_embeds"].unsqueeze(2)

    if reduce == "mean":
        outputs = {}
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mass = (torch.bmm(refs_z, preds_z).squeeze() >= threshold).float().mean()
            outputs[f"semantic_similarity@{threshold}"] = mass.item()
        outputs["semantic_similarity@avg"] = torch.Tensor(list(outputs.values())).mean().item()
        return outputs

    # Return without reduction
    outputs = {}
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mass = torch.bmm(refs_z, preds_z).squeeze() >= threshold
        outputs[f"semantic_similarity@{threshold}"] = mass.int().tolist()
    outputs["semantic_similarity@avg"] = torch.Tensor(list(outputs.values())).mean(dim=0).tolist()
    return outputs


@register_aggregation("median", can_bootstrap=True)
def median(arr: list) -> float:
    """Calculate the median of a list of numbers.

    Args:
    ----
        arr (list): List of numbers.

    """
    return arr[len(arr) // 2]


@register_aggregation("perplexity")
def perplexity(items: list) -> float:
    """Calculate the perplexity on a list of documents.

    Note: perplexity must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    return math.exp(-mean(items))


@register_aggregation("semantic_similarity")
def semantic_similarity(
    items: list, reduce: Literal["none", "mean"] = "mean"
) -> float | list[float]:
    """Calculate the semantic similarity on a list of documents.

    Note: semantic similarity can be calculated independently per each document in a benchmark.
    However, we evaluate it as a group-level metric as we can batch samples in the forward to
    the SentenceBERT model to speed up evaluation. For this reason, we define it as an aggregation
    metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.
        reduce ("none" | "mean"): Reduction operation on the values. When "none", returns the
            value per sample; when "mean", returns the mean similarity. Defaults to "mean".

    """
    from src.data.pipelines.text import encode_sentence_bert

    if reduce not in ["none", "mean"]:
        raise ValueError(
            "Unknown `reduce` value for `semantic_similarity` metric."
            ' Expected "none" or "mean", but got "%s"',
            reduce,
        )

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]

    refs = [ref[0] if isinstance(ref, list) else ref for ref in refs]
    preds = [pred[-1] if isinstance(pred, list) else pred for pred in preds]

    data = datasets.Dataset.from_dict({"prediction": preds, "reference": refs})
    data.set_format("torch")

    data = data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "reference"},
    )
    data = data.map(
        encode_sentence_bert,
        batched=True,
        batch_size=1024,
        fn_kwargs={"input_column": "prediction"},
    )

    refs_z = data["reference_sentence_bert_embeds"].unsqueeze(1)
    preds_z = data["prediction_sentence_bert_embeds"].unsqueeze(2)

    if reduce == "mean":
        return torch.bmm(refs_z, preds_z).squeeze().mean().item()

    # Return without reduction
    return torch.bmm(refs_z, preds_z).squeeze().tolist()


@register_aggregation("ter")
def ter(items: list) -> float:
    """Calculate the Translation-Error-Rate score on a list of documents.

    Note: TER score must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    import sacrebleu

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


@register_aggregation("textual_inclusion_llama32")
def textual_inclusion_llama32(
    items: list, reduce: Literal["none", "mean"] = "mean"
) -> float | list[float]:
    """Calculate the textual inclusion score with Llama 3.2 on a list of documents.

    Note: exact match with Llama 3.2 can be calculated independently per each document in a
    benchmark. However, we evaluate it as a group-level metric as we can batch samples in the
    forward to the Llama model to speed up evaluation. For this reason, we define it as an
    aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.
        reduce ("none" | "mean"): Reduction operation on the values. When "none", returns the
            value per sample; when "mean", returns the mean similarity. Defaults to "mean".

    """
    from src.data.pipelines.text import textual_inclusion_llama32 as _textual_inclusion_llama32

    if reduce not in ["none", "mean"]:
        raise ValueError(
            "Unknown `reduce` value for `textual_inclusion_llama32` metric."
            ' Expected "none" or "mean", but got "%s"',
            reduce,
        )

    refs = list(zip(*items, strict=True))[0]
    preds = list(zip(*items, strict=True))[1]

    refs = [ref[0] if isinstance(ref, list) else ref for ref in refs]
    preds = [pred[-1] if isinstance(pred, list) else pred for pred in preds]

    data = datasets.Dataset.from_dict({"prediction": preds, "reference": refs})
    scores = data.map(
        _textual_inclusion_llama32,
        batch_size=1024,
        batched=True,
    )["exact_match_score"]
    scores = [int(score) if score in ["0", "1"] else 0 for score in scores]

    if reduce == "mean":
        return np.mean(scores)

    # Return without reduction
    return scores


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items: list) -> float:
    """Calculate the weighted perplexity on a list of documents.

    Note: perplexity must be calculated across all documents in a benchmark. For simplicity,
    we define it as an aggregation metric and pair it with a no-op passthrough metric function.

    Args:
    ----
        items (list): List of documents.

    """
    return math.exp(-_weighted_mean(items))
