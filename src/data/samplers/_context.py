from random import Random

import datasets

from src.data.samplers._api import register_sampler
from src.data.samplers._base import Sampler
from src.data.tasks import Task

__all__ = ["CONTEXT_SAMPLERS", "ContextSampler", "FirstNSampler"]

CONTEXT_SAMPLERS = ["default", "first_n"]


@register_sampler("default")
class ContextSampler(Sampler):
    """Sampler for few-shot context examples.

    Args:
    ----
        docs (list): List of documents.
        task (Task): The task object.
        fewshot_indices (list, optional): List of indices of fewshot examples. Defaults to None.
        rnd (Random): Random object. Defaults to None.

    """

    def __init__(
        self,
        docs: datasets.Dataset,
        task: Task,
        fewshot_indices: list | None = None,
        rnd: Random | None = None,
    ) -> None:
        if not rnd:
            raise ValueError("must pass rnd to FewShotSampler!")

        self.rnd = rnd
        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc: dict, num_fewshot: int) -> str:
        """Get the context for the task.

        Args:
        ----
            doc (dict): The document.
            num_fewshot (int): The number of fewshot examples.

        """
        n_samples = num_fewshot

        # Draw an extra sample if using same split as evaluating on
        if self.config.fewshot_split == self.config.test_split:
            n_samples += 1

        # Draw `n_samples` docs from fewshot_docs
        fewshot_examples = self.sample(n_samples)

        # Get rid of the doc that we are evaluating on, if it's in the fewshot
        # TODO should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshot_examples if x != doc][:num_fewshot]

        # TODO is separating doc_to_text and doc_to_target by one space always desired?
        labeled_examples = []
        for _doc in selected_docs:
            first_part, second_part = None, None

            if self.config.doc_to_choice is None or isinstance(self.doc_to_text(_doc), str):
                first_part = self.doc_to_text(_doc)
            else:
                first_part = self.doc_to_choice(_doc)[self.doc_to_text(_doc)]

            if isinstance(self.doc_to_target(_doc), list):
                second_part = str(self.doc_to_target(_doc)[0])
            elif self.config.doc_to_choice is None or isinstance(self.doc_to_target(_doc), str):
                second_part = self.doc_to_target(_doc)
            else:
                second_part = str(self.doc_to_choice(_doc)[self.doc_to_target(_doc)])

            labeled_examples.append(first_part + self.target_delimiter + second_part)

        labeled_examples = self.fewshot_delimiter.join(labeled_examples) + self.fewshot_delimiter

        return labeled_examples

    def sample(self, n_samples: int) -> list:
        """Draw `n` samples from the fewshot docs.

        Args:
        ----
            n_samples (int): The number of samples to draw.

        """
        return self.rnd.sample(self.docs, n_samples)


@register_sampler("first_n")
class FirstNSampler(ContextSampler):
    """Sampler to select the first N samples to create the context.

    Args:
    ----
        docs (list): List of documents.
        task (Task): The task object.
        fewshot_indices (list, optional): List of indices of fewshot examples. Defaults to None.
        rnd (Random): Random object. Defaults to None.

    """

    def sample(self, n_samples: int) -> list:
        """Draw the first `n` samples in order from the specified split.

        Args:
        ----
            n_samples (int): The number of samples to draw.

        """
        if n_samples > len(self.docs):
            raise ValueError(
                f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are"
                " available."
            )

        return self.docs[:n_samples]
