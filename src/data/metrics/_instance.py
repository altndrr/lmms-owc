import re
import string

import numpy as np

from src.data.metrics._api import register_metric

__all__ = [
    "INSTANCE_METRICS",
    "acc",
    "acc_all",
    "acc_mutual_info",
    "acc_norm",
    "anls",
    "bits_per_byte",
    "bleu",
    "brier_score",
    "bypass",
    "byte_perplexity",
    "chrf",
    "concept_semantic_similarity",
    "exact_match",
    "f1",
    "mcc",
    "mean_average_semantic_similarity",
    "perplexity",
    "semantic_similarity",
    "ter",
    "textual_inclusion",
    "textual_inclusion_llama32",
    "word_perplexity",
]

INSTANCE_METRICS = ["acc_all", "ansl", "exact_match", "textual_inclusion"]


@register_metric(
    group_fn_name="mean",
    higher_is_better=True,
    output_types=["loglikelihood", "multiple_choice"],
)
def acc(items: list) -> list:
    """Calculate the accuracy on a list of documents.

    Note: this is a passthrough function used alongside the `mean` aggregation function
    since the accuracy must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(group_fn_name="mean", higher_is_better=True, output_types=["loglikelihood"])
def acc_all(items: list) -> float:
    """Calculate the accuracy on a list of documents.

    Args:
    ----
        items (list): List of documents.

    """
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items, strict=True))[0]
    docs = list(zip(*items, strict=True))[1]

    for doc, pred in zip(docs, preds, strict=True):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


@register_metric(group_fn_name="mean", higher_is_better=True, output_types=["multiple_choice"])
def acc_mutual_info(items: list) -> list:
    """Calculate the mutual information on a list of documents.

    Note: this is a passthrough function used alongside the `mean` aggregation function
    since the mutual information must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="mean",
    higher_is_better=True,
    output_types=["loglikelihood", "multiple_choice"],
)
def acc_norm(items: list) -> list:
    """Calculate the normalized accuracy on a list of documents.

    Note: this is a passthrough function used alongside the `mean` aggregation function
    since the normalized accuracy must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings.

    Args:
    ----
        s1 (str): The first string.
        s2 (str): The second string.

    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


@register_metric(higher_is_better=True, output_types=["generate_until"], group_fn_name="mean")
def anls(references: list, predictions: list, threshold: float = 0.5) -> dict:
    """Calculate the ANLS score on a list of documents.

    Args:
    ----
        references (list): List of references.
        predictions (list): List of predictions.
        threshold (float, optional): The threshold value. Defaults to 0.5.

    """
    values = []

    # Unwrap predictions if it's a nested list
    pred = predictions[0] if isinstance(predictions[0], str) else predictions[0][0]

    for answer in references:
        # Preprocess both the answers - gt and prediction
        gt_answer = " ".join(answer.strip().lower().split())
        det_answer = " ".join(pred.strip().lower().split())

        dist = _levenshtein_distance(gt_answer, det_answer)
        length = max(len(answer.upper()), len(pred.upper()))
        values.append(0.0 if length == 0 else float(dist) / float(length))

    question_result = 1 - min(values)

    if question_result < threshold:
        question_result = 0
    return {"anls": question_result}


@register_metric(
    group_fn_name="bits_per_byte",
    higher_is_better=False,
    output_types=["loglikelihood_rolling"],
)
def bits_per_byte(items: list) -> list:
    """Calculate the bits per byte on a list of documents.

    Note: this is a passthrough function used alongside the `bits_per_byte` aggregation function
    since the bits per byte must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="bleu",
    higher_is_better=True,
    output_types=["generate_until", "generate_until_multi_round"],
    can_bootstrap=True,
)
def bleu(items: list) -> list:
    """Calculate the BLEU score on a list of documents.

    Note: this is a passthrough function used alongside the `bleu` aggregation function
    since the BLEU score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="brier_score",
    higher_is_better=False,
    output_types=["multiple_choice"],
)
def brier_score(items: list) -> list:
    """Calculate the Brier score on a list of documents.

    Note: this is a passthrough function used alongside the `brier_score` aggregation function
    since the Brier score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="bypass",
    higher_is_better=True,
    output_types=[
        "loglikelihood",
        "multiple_choice",
        "generate_until",
        "generate_until_multi_round",
    ],
)
def bypass(items: list) -> list:
    """Calculate the bypass score on a list of documents.

    Note: this is a passthrough function used alongside the `bypass` aggregation function
    since the bypass score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="weighted_perplexity",
    higher_is_better=False,
    output_types=["loglikelihood_rolling"],
)
def byte_perplexity(items: list) -> list:
    """Calculate the byte perplexity on a list of documents.

    Note: this is a passthrough function used alongside the `weighted_perplexity` aggregation
    function since the byte perplexity must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="chrf",
    higher_is_better=True,
    output_types=["generate_until", "generate_until_multi_round"],
    can_bootstrap=True,
)
def chrf(items: list) -> list:
    """Calculate the chrF score on a list of documents.

    Note: this is a passthrough function used alongside the `chrf` aggregation function
    since the chrF score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="concept_semantic_similarity",
    higher_is_better=True,
    output_types=["generate_until"],
    can_bootstrap=False,
)
def concept_semantic_similarity(items: list) -> list:
    """Calculate the semantic similarity of the response concepts on a list of documents.

    Note: this is a passthrough function used alongside the `concept_semantic_similarity`
    aggregation function since the semantic similarity metrics can be batched to speed up its
    evaluation.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(group_fn_name="mean", higher_is_better=True, output_types=["generate_until"])
def exact_match(
    predictions: list,
    references: list,
    regexes_to_ignore: list | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> dict:
    """Calculate the exact match score on a list of documents.

    Args:
    ----
        predictions (list): List of predictions.
        references (list): List of references.
        regexes_to_ignore (list, optional): List of regexes to ignore. Defaults to None.
        ignore_case (bool, optional): Whether to ignore case. Defaults to False.
        ignore_punctuation (bool, optional): Whether to ignore punctuation. Defaults to False.
        ignore_numbers (bool, optional): Whether to ignore numbers. Defaults to False.

    """
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


@register_metric(
    group_fn_name="f1",
    higher_is_better=True,
    output_types=["multiple_choice"],
    can_bootstrap=True,
)
def f1(items: list) -> list:
    """Calculate the F1 score on a list of documents.

    Note: this is a passthrough function used alongside the `f1` aggregation function
    since the F1 score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="matthews_corrcoef",
    higher_is_better=True,
    output_types=["multiple_choice"],
    can_bootstrap=True,
)
def mcc(items: list) -> list:
    """Calculate the Matthews correlation coefficient on a list of documents.

    Note: this is a passthrough function used alongside the `matthews_corrcoef` aggregation
    function since the Matthews correlation coefficient must be calculated across all documents
    in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="mean_average_semantic_similarity",
    higher_is_better=True,
    output_types=["generate_until"],
    can_bootstrap=False,
)
def mean_average_semantic_similarity(items: list) -> list:
    """Calculate the mean average semantic similarity on a list of documents.

    Note: this is a passthrough function used alongside the `mean_average_semantic_similarity`
    aggregation function since the semantic similarity metrics can be batched to speed up its
    evaluation.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="perplexity",
    higher_is_better=False,
    output_types=["loglikelihood"],
    can_bootstrap=True,
)
def perplexity(items: list) -> list:
    """Calculate the perplexity on a list of documents.

    Note: this is a passthrough function used alongside the `perplexity` aggregation function
    since the perplexity must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="semantic_similarity",
    higher_is_better=True,
    output_types=["generate_until"],
    can_bootstrap=False,
)
def semantic_similarity(items: list) -> list:
    """Calculate the semantic similarity on a list of documents.

    Note: this is a passthrough function used alongside the `semantic_similarity` aggregation
    function since the semantic similarity metrics can be batched to speed up its evaluation.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(group_fn_name="mean", higher_is_better=True, output_types=["generate_until"])
def textual_inclusion(predictions: list, references: list) -> dict:
    """Calculate if the prediction contains the answer on a list of documents.

    Args:
    ----
        predictions (list): List of predictions.
        references (list): List of references.

    """
    score_list = [
        ref.lower().strip() in pred.lower().strip()
        for ref, pred in zip(references, predictions, strict=True)
    ]

    return {"textual_inclusion": np.mean(score_list)}


@register_metric(
    group_fn_name="textual_inclusion_llama32",
    higher_is_better=True,
    output_types=["generate_until"],
    can_bootstrap=False,
)
def textual_inclusion_llama32(items: list) -> list:
    """Calculate the exact match score with Llama 3.2 on a list of documents.

    Note: this is a passthrough function used alongside the `textual_inclusion_llama32` aggregation
    function since the metric can be batched to speed up its evaluation.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="ter",
    higher_is_better=True,
    output_types=["generate_until", "generate_until_multi_round"],
    can_bootstrap=True,
)
def ter(items: list) -> list:
    """Calculate the TER score on a list of documents.

    Note: this is a passthrough function used alongside the `ter` aggregation function
    since the TER score must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items


@register_metric(
    group_fn_name="weighted_perplexity",
    higher_is_better=False,
    output_types=["loglikelihood_rolling"],
)
def word_perplexity(items: list) -> list:
    """Calculate the word perplexity on a list of documents.

    Note: this is a passthrough function used alongside the `weighted_perplexity` aggregation
    function since the word perplexity must be calculated across all documents in a benchmark.

    Args:
    ----
        items (list): List of documents.

    """
    return items
