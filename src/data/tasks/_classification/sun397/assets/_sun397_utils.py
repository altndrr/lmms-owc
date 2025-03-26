from pathlib import Path

import datasets
import pandas as pd
from PIL import Image

from src import utils

__all__ = ["doc_to_text", "doc_to_text_multi_round", "doc_to_visual", "doc_to_target", "download"]


def doc_to_text(doc: dict, model_specific_kwargs: dict) -> str:
    """Process SUN397 document to extract question text.

    Args:
    ----
        doc (dict): Document containing the sample information.
        model_specific_kwargs (dict): Extra kwargs about the repository.

    """
    pre_prompt = model_specific_kwargs.get("pre_prompt", "")
    prompt = model_specific_kwargs.get("prompt", "What's in the image?")
    post_prompt = model_specific_kwargs.get("post_prompt", "")

    return pre_prompt + prompt + post_prompt


def doc_to_text_multi_round(
    doc: dict,
    model_specific_kwargs: dict,
    round_idx: int | None = None,
    previous_round_results: list | None = None,
    last_round_info: dict | None = None,
) -> str | tuple:
    """Process SUN397 document to extract information for multi-round dialogue.

    Args:
    ----
        doc (dict): Document containing the sample information.
        model_specific_kwargs (dict): Extra kwargs about the model.
        round_idx (int, optional): Index of the current dialogue round. Defaults to None.
        previous_round_results (list, optional): Results from all the previous dialogue rounds.
            Defaults to None.
        last_round_info (dict, optional): Information from the last dialogue round. Defaults
            to None.

    """
    visual, text = None, None
    should_terminate = False

    if previous_round_results is None:
        previous_round_results = []

    # Prepare query information
    pre_prompt = model_specific_kwargs.get("pre_prompt", "")
    post_prompt = model_specific_kwargs.get("post_prompt", "")
    prompts = model_specific_kwargs.get("prompts")

    if not isinstance(prompts, list) or len(prompts) < 2:
        raise ValueError("`multi_round` expects at least two questions")

    if round_idx is None:
        return pre_prompt + prompts[0] + post_prompt

    if round_idx < len(prompts):
        visual = None
        text = pre_prompt + prompts[round_idx] + post_prompt
    else:
        should_terminate = True

    return visual, text, should_terminate, previous_round_results, last_round_info


def doc_to_visual(doc: dict) -> list:
    """Convert SUN397 document to visual format.

    Args:
    ----
        doc (dict): Document containing the sample information.

    """
    return [Image.open(doc["visual"]).convert("RGB")]


def doc_to_target(doc: dict) -> str:
    """Convert SUN397 document to target format.

    Args:
    ----
        doc (dict): Document containing the sample information.

    """
    return doc["target"].replace("_", " ")


def download(output_dir: str = "data", cache_dir: str = ".cache") -> datasets.DatasetDict:
    """Download SUN397 dataset from the web and convert it to the HuggingFace format.

    Args:
    ----
        output_dir (str): The path where to save the HuggingFace dataset. Default to "data".
        cache_dir (str): The path where to save cache files. Default to ".cache".

    """
    data_url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    dataset_path = Path(cache_dir, "data", "SUN397")
    output_path = Path(output_dir) / "sun397"

    if output_path.exists():
        return

    # Download the dataset
    if not dataset_path.exists():
        target_path = Path(cache_dir, "data", Path(data_url).name)
        utils.download_data(data_url, target_path, from_gdrive=False)
        utils.extract_data(target_path)

    # Read the dataset
    metadata_fp = Path(__file__).parent / "metadata.csv"
    split_fp = Path(__file__).parent / "split_coop.csv"

    metadata_df = pd.read_csv(metadata_fp)
    class_names = metadata_df["class_name"].tolist()
    classes_to_idx = {str(c): i for i, c in enumerate(class_names)}

    split_df = pd.read_csv(split_fp)
    data = datasets.DatasetDict()
    for split in ["train", "val", "test"]:
        image_paths = split_df[split_df["split"] == split]["filename"]
        image_paths = image_paths.apply(lambda x: str(dataset_path / x)).tolist()
        folder_names = []
        for f in image_paths:
            parent = Path(f).parent
            if len(parent.parent.name) != 1:
                folder_names.append(f"{parent.name} {parent.parent.name}")
            else:
                folder_names.append(parent.name)
        labels = [classes_to_idx[c] for c in folder_names]
        data[split] = utils.load_image_folder_as_hf_dataset(
            str(dataset_path), images=image_paths, labels=labels, class_names=class_names
        )

    data.save_to_disk(output_path)
