import re
from typing import Any

from PIL.Image import Image

from src.data.filters._extraction import MultiChoiceRegexFilter

__all__ = [
    "CustomMultiChoiceRegexFilter",
    "aggregate_muirbench_score",
    "doc_to_text",
    "doc_to_visual",
    "doc_to_target",
    "process_results",
]


class CustomMultiChoiceRegexFilter(MultiChoiceRegexFilter):
    """Filter for processing multiple choice responses using regex.

    Args:
    ----
        resps (list): List of model responses
        docs (list): List of corresponding documents

    """

    def apply(self, resps: list[list[str]], docs: list[dict[str, Any]]) -> list[str]:
        """Apply filtering to multiple choice responses.

        Args:
        ----
            resps (list): List of model responses to filter
            docs (list): List of corresponding documents

        """
        filtered_resps = []

        for r, _ in zip(resps, docs, strict=False):
            option_letter_regex = re.compile(r"^\s*([A-Z])\.")

            filtered = []
            for resp in r:
                match = option_letter_regex.match(resp)
                if match:
                    filtered.append(match.group(1))
                else:
                    filtered.append(resp)

            filtered_resps.append(filtered[0])

        return filtered_resps


def aggregate_muirbench_score(results: list[dict]) -> float:
    """Aggregate results and calculates scores per task and overall.

    Args:
    ----
        results (list): List of dictionaries containing prediction results and metadata.
            Each dictionary should have a nested structure with task information,
            predictions, and answers.

    """
    task_num = {}
    score = 0
    task_score = {}
    for result in results:
        if result["task"] not in task_score:
            task_score[result["task"]] = 0

        if result["task"] not in task_num:
            task_num[result["task"]] = 0

        if result["pred"].lower().strip() == result["answer"].lower().strip():
            task_score[result["task"]] += 1
            score += 1
        task_num[result["task"]] += 1

    score = score / len(results)
    task_score = {k: v / task_num[k] for k, v in task_score.items()}

    return score


def doc_to_text(doc: dict[str, Any], model_specific_kwargs: dict[str, str] | None = None) -> str:
    """Convert a MUIR document to text format with question and multiple choice options."""
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = model_specific_kwargs["post_prompt"]
    pre_prompt = model_specific_kwargs["pre_prompt"]
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join(
        [f"{option}. {choice}" for option, choice in zip(options, choices, strict=True)]
    )
    return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"


def doc_to_visual(doc: dict[str, Any]) -> list[Image]:
    """Convert image list from a MUIR document to RGB format.

    Args:
    ----
        doc (dict): A dictionary containing MUIR document data with image information.

    """
    image_list = [image.convert("RGB") for image in doc["image_list"]]
    return image_list


def doc_to_target(doc: dict[str, Any]) -> str:
    """Extract the target answer from a MUIR document.

    Args:
    ----
        doc (dict): A dictionary containing MUIR document data with target answer.

    """
    return doc["answer"]


def process_results(doc: dict[str, Any], result: list[str]) -> dict[str, dict[str, Any]]:
    """Process prediction results and creates a score dictionary with metadata.

    Args:
    ----
        doc: A dictionary containing MUIR document data.
        result: A list of prediction strings to be processed.

    """
    pred = result[0]
    task = doc["task"]
    idx = doc["idx"]
    image_relation = doc["image_relation"]
    answer = doc["answer"]
    image_type = doc["image_type"]

    data_dict = {
        "pred": pred,
        "task": task,
        "idx": idx,
        "image_relation": image_relation,
        "answer": answer,
        "image_type": image_type,
    }

    return {"muirbench_score_overall": data_dict}
