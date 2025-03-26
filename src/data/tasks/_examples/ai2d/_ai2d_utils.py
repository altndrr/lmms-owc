import re

from src.data.filters._extraction import MultiChoiceRegexFilter

__all__ = ["CustomMultiChoiceRegexFilter", "doc_to_text", "doc_to_visual", "doc_to_target"]


class CustomMultiChoiceRegexFilter(MultiChoiceRegexFilter):
    """Filter for processing multiple choice responses using regex.

    Args:
    ----
        resps (list): List of model responses
        docs (list): List of corresponding documents

    """

    def apply(self, resps: list, docs: list) -> list:
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


def doc_to_text(doc: dict, model_specific_kwargs: dict) -> str:
    """Convert AI2D document to text format.

    Args:
    ----
        doc (dict): Document containing question and options
        model_specific_kwargs (dict): Kwargs containing prompt format and text

    """
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    post_prompt = model_specific_kwargs["post_prompt"]
    pre_prompt = model_specific_kwargs["pre_prompt"]
    if model_specific_kwargs["prompt_format"] == "mcq":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = "\n".join(
            [f"{option}. {choice}" for option, choice in zip(options, choices, strict=True)]
        )
        return f"{pre_prompt}{question}\n{choices_str}{post_prompt}"
    elif model_specific_kwargs["prompt_format"] == "qa":
        options = "\n".join(choices)
        return f"{pre_prompt}{question}{options}{post_prompt}"
    elif model_specific_kwargs["prompt_format"] == "mcq_xcomposer":
        options = [chr(ord("A") + i) for i in range(len_choices)]
        choices_str = " ".join(
            [f"{option}. {choice}" for option, choice in zip(options, choices, strict=True)]
        )
        return f"{pre_prompt}{question}\nContext: N/A\n{choices_str}{post_prompt}"
    else:
        raise ValueError(f"Unknown prompt format: {model_specific_kwargs['prompt_format']}")


def doc_to_visual(doc: dict) -> list:
    """Convert AI2D document to visual format.

    Args:
    ----
        doc (dict): Document containing image data

    """
    return [doc["image"].convert("RGB")]


def doc_to_target(doc: dict, model_specific_target_kwargs: str) -> str:
    """Convert AI2D document to target format.

    Args:
    ----
        doc (dict): Document containing options and answer
        model_specific_target_kwargs (str): Format specification for target output

    """
    if model_specific_target_kwargs == "mcq":
        len_choices = len(doc["options"])
        options = [chr(ord("A") + i) for i in range(len_choices)]
        return options[int(doc["answer"])]
    elif model_specific_target_kwargs == "qa":
        return doc["options"][int(doc["answer"])]
    else:
        raise ValueError("Unknown target kwargs for ai2d task.")
