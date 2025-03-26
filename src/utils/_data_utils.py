import hashlib
import importlib.util
import os
import re
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import datasets
import dill
import gdown
import numpy as np
import requests  # pytype: disable=pyi-error
import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined
from pytablewriter import LatexTableWriter, MarkdownTableWriter
from tqdm import tqdm
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from src.utils._logging_utils import get_logger

__all__ = [
    "HIGHER_IS_BETTER_SYMBOLS",
    "apply_jinja_template",
    "convert_non_serializable",
    "delete_cache",
    "download_data",
    "extract_data",
    "get_task_datetime_from_filename",
    "get_task_name_from_filename",
    "get_results_filenames",
    "get_sample_results_filenames",
    "load_from_cache",
    "load_image_folder_as_hf_dataset",
    "load_yaml_config",
    "make_table",
    "save_to_cache",
]

log = get_logger(__name__, rank_zero_only=True)

# Cache constants
CACHE_HASH_INPUT = "EleutherAI-lm-evaluation-harness"
CACHE_HASH_PREFIX = hashlib.sha256(CACHE_HASH_INPUT.encode("utf-8")).hexdigest()
CACHE_FILE_SUFFIX = f".{CACHE_HASH_PREFIX}.pickle"

# Cache path constants
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = os.getenv("CACHE_DIR", None) or f"{WORKING_DIR}/.cache"

# Metrics constants
HIGHER_IS_BETTER_SYMBOLS = {True: "↑", False: "↓"}


def _regex_replace(string: str, pattern: str, repl: str, count: int = 0) -> str:
    """Implement the `re.sub` function as a custom Jinja filter.

    Args:
    ----
        string (str): The string to be processed.
        pattern (str): The pattern to be replaced.
        repl (str): The replacement string.
        count (int, optional): The number of replacements to make. Defaults to 0.

    """
    return re.sub(pattern, repl, string, count=count)


env = Environment(loader=BaseLoader, undefined=StrictUndefined, autoescape=True)
env.filters["regex_replace"] = _regex_replace


def apply_jinja_template(template: str, doc: dict) -> str:
    """Apply a Jinja template to a document.

    Args:
    ----
        template (str): The template to be applied.
        doc (dict): The document to apply the template

    """
    r_template = env.from_string(template)
    return r_template.render(**doc)


def convert_non_serializable(obj: Any) -> int | str | list:  # noqa: ANN401
    """Convert non-serializable objects into serializable types.

    Args:
    ----
        obj (Any): The object to be converted.

    """
    if isinstance(obj, (np.int64 | np.int32)):
        return int(obj)
    elif isinstance(obj, set):
        return list(obj)

    return str(obj)


def delete_cache(key: str | None = None) -> None:
    """Delete cached files from the cache directory.

    Args:
    ----
        key (str, optional): The key to be used to delete the cache. Defaults to None.

    """
    if key is None:
        key = ""

    files = Path(CACHE_DIR).rglob("*")
    for file in files:
        if file.name.startswith(key) and file.name.endswith(CACHE_FILE_SUFFIX):
            file.unlink()


def download_data(url: str, target: Path, from_gdrive: bool = False) -> None:
    """Download data from a URL.

    Args:
    ----
        url (str): The URL to download the data from.
        target (Path): The path to save the data to.
        from_gdrive (bool): Whether the data is from Google Drive.

    """
    if not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=False)

    if from_gdrive:
        url += "&confirm=t"  # Append the confirm parameter to the URL
        gdown.download(url, str(target), quiet=False)
    else:
        header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
        with requests.get(url, stream=True, timeout=10.0, headers=header) as r:
            r.raise_for_status()
            chunk_size = 8192
            pbar = tqdm(
                r.iter_content(chunk_size=chunk_size),
                total=int(r.headers["Content-Length"]) / chunk_size,
                desc=f"Downloading data to {target}",
            )
            with open(target, "wb") as f:
                for chunk in pbar:
                    f.write(chunk)


def extract_data(target: Path) -> None:
    """Extract data from an archive.

    Args:
    ----
        target (Path): The path to the file to extract.

    """
    if target.name.endswith(".zip"):
        with zipfile.ZipFile(target, "r") as zip_ref:
            # Validate file paths before extraction
            for file in zip_ref.filelist:
                # Ensure the normalized path doesn't escape the target directory
                target_path = (target.parent / file.filename).resolve()
                if not str(target_path).startswith(str(target.parent.resolve())):
                    raise ValueError(f"Attempted path traversal in zip file: {file}")
                zip_ref.extract(file, target.parent)
    elif target.name.endswith(".tar"):
        with tarfile.open(target, "r") as tar_ref:
            for member in tar_ref.getmembers():
                # Ensure the normalized path doesn't escape the target directory
                target_path = (target.parent / member.name).resolve()
                if not str(target_path).startswith(str(target.parent.resolve())):
                    raise ValueError(f"Attempted path traversal in tar file: {member.name}")
                tar_ref.extract(member, target.parent)
    elif target.name.endswith(".tar.gz") or target.name.endswith(".tgz"):
        with tarfile.open(target, "r:gz") as tar_ref:
            for member in tar_ref.getmembers():
                # Ensure the normalized path doesn't escape the target directory
                target_path = (target.parent / member.name).resolve()
                if not str(target_path).startswith(str(target.parent.resolve())):
                    raise ValueError(f"Attempted path traversal in tar file: {member.name}")
                tar_ref.extract(member, target.parent)
    else:
        raise NotImplementedError(f"Unsupported file format: {target.suffix}")


def get_task_name_from_filename(filename: str) -> str:
    """Get the task name from its filename.

    Args:
    ----
        filename (str): The filename to extract the task name from.

    """
    return filename[filename.find("_") + 1 : filename.rfind("_")]


def get_task_datetime_from_filename(filename: str) -> str:
    """Get the task datetime from its filename.

    Args:
    ----
        filename (str): The filename to extract the datetime from.

    """
    return filename[filename.rfind("_") + 1 :].replace(".jsonl", "")


def get_results_filenames(filenames: list[str]) -> list[str]:
    """Extract filenames that correspond to aggregated results.

    Args:
    ----
        filenames (list[str]): The list of filenames to extract the results from.

    """
    return [f for f in filenames if "results" in f and ".json" in f]


def get_sample_results_filenames(filenames: list[str]) -> list[str]:
    """Extract filenames that correspond to sample results.

    Args:
    ----
        filenames (list[str]): The list of filenames to extract the sample results from.

    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def load_from_cache(file_name: str) -> dict | None:
    """Load a cached file from the cache directory.

    Args:
    ----
        file_name (str): The name of the file to be loaded.

    """
    path = f"{CACHE_DIR}/{file_name}{CACHE_FILE_SUFFIX}"
    try:
        with open(path, "rb") as file:
            cached_task_dict = dill.loads(file.read())  # noqa: S301
            return cached_task_dict
    except FileNotFoundError:
        log.debug("`%s` is not cached, generating...", file_name)


def load_image_folder_as_hf_dataset(
    root: str,
    images: list[str] | None = None,
    labels: list[int] | list[list[int]] | None = None,
    class_names: list[str] | None = None,
    classes_to_idx: dict[str, int] | None = None,
) -> datasets.Dataset:
    """Load an ImageFolder-like dataset as a HuggingFace dataset.

    Args:
    ----
        root (str): The root directory of the dataset.
        images (list[str] | None, optional): List of image file paths. Defaults to None.
        labels (list[int] | list[list[int]] | None, optional): List of labels or list of label
            lists. Defaults to None.
        class_names (list[str] | None, optional): List of class names. Defaults to None.
        classes_to_idx (dict[str, int] | None, optional): Dictionary mapping class names to
            indices. Defaults to None.

    """
    if not images:
        images = [str(path) for path in Path(root).glob("*/*")]

    if not class_names:
        class_names = {Path(f).parent.name for f in images}

    if not labels:
        folder_names = {Path(f).parent.name for f in images}
        folder_names = sorted(folder_names)
        folder_names_to_idx = {c: i for i, c in enumerate(folder_names)}
        labels = [folder_names_to_idx[Path(f).parent.name] for f in images]

    classes_to_idx = classes_to_idx or {c: i for i, c in enumerate(class_names)}
    target_names = [class_names[t] for t in labels]

    data = datasets.Dataset.from_dict({"visual": images, "target": target_names})

    return data


def _ignore_constructor(loader: yaml.Loader, node: yaml.Node) -> yaml.Node:
    """Ignore the constructor for the YAML loader.

    Args:
    ----
        loader (yaml.Loader): The YAML loader.
        node (yaml.Node): The node to be processed.

    """
    return node


def _import_function(loader: yaml.Loader, node: yaml.nodes.ScalarNode) -> Callable:
    """Import a function from a module.

    Args:
    ----
        loader (yaml.Loader): The YAML loader.
        node (yaml.nodes.Node): The node to be processed.

    """
    function_name = str(loader.construct_scalar(node))
    yaml_path = os.path.dirname(loader.name)

    (*module_name, function_name) = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, f"{module_name}.py"))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ValueError("Module specification cannot be None")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(
    yaml_path: str | None,
    yaml_config: dict | None = None,
    yaml_dir: str | None = None,
    mode: Literal["simple", "full"] = "full",
) -> dict:
    """Load a YAML configuration file.

    Args:
    ----
        yaml_path (str, optional): The path to the YAML file.
        yaml_config (dict, optional): The YAML configuration. Defaults to None.
        yaml_dir (str, optional): The directory of the YAML file. Defaults to None.
        mode (Literal["simple", "full"], optional): The mode to load the YAML file. Defaults
            to "full".

    """
    if mode == "simple":
        constructor_fn = _ignore_constructor
    elif mode == "full":
        constructor_fn = _import_function
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Add the _import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    if yaml_dir is None:
        raise ValueError("yaml_dir is None")
    if yaml_config is None:
        raise ValueError("yaml_config is None")

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path. If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not Path(path).is_file():
                path = Path(yaml_dir) / path

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except (OSError, ParserError, ScannerError) as e:
                raise e
        final_yaml_config.update(yaml_config)
        return final_yaml_config

    return yaml_config


def make_table(
    results: dict, col: Literal["results", "groups"] = "results", sort: bool = False
) -> str:
    """Generate table of results.

    Args:
    ----
        results (dict): The dictionary of results.
        col (Literal["results", "groups"]): The column to generate the table for. Defaults to
            "results".
        sort (bool): Whether to sort the results. Defaults to False.

    """
    if col == "results":
        column_name = "Tasks"
    elif col == "groups":
        column_name = "Groups"
    else:
        raise ValueError(f"Invalid column name: {col}")

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []
    keys = results[col].keys()

    # Sort entries alphabetically by task or group name.
    # Note that we skip sorting by default as order matters for multi-level table printing.
    if sort:
        keys = sorted(keys)

    for k in keys:
        dic = results[col][k]
        version = results["versions"].get(k, "    N/A")
        n = str(results["n-shot"].get(k, " "))
        higher_is_better = results.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = f"{v:.4f}" if isinstance(v, float) else v
            if v == "" or v is None:
                v = "N/A"

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                se = "   N/A" if se == "N/A" or se == [] else f"{se:.4f}"
                if v != []:
                    values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, v, "", ""])

    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # TODO beautify the return table
    return md_writer.dumps()


def save_to_cache(file_name: str, obj: list) -> None:
    """Save an object to the cache directory.

    Args:
    ----
        file_name (str): The name of the file to be saved.
        obj (list): The object to be saved.

    """
    cache_dir = Path(CACHE_DIR)
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    file_path = cache_dir / f"{file_name}{CACHE_FILE_SUFFIX}"

    serializable_obj = []
    for item in obj:
        sub_serializable_obj = []
        for subitem in item:
            # Handle the arguments since doc_to_visual is callable method and not serializable
            if hasattr(subitem, "arguments"):
                args = subitem.arguments
                serializable_args = tuple(arg if not callable(arg) else None for arg in args)
                subitem.arguments = serializable_args
            sub_serializable_obj.append(convert_non_serializable(subitem))
        serializable_obj.append(sub_serializable_obj)

    log.debug("Saving %s to cache...", file_path)
    with open(file_path, "wb") as file:
        file.write(dill.dumps(serializable_obj))
