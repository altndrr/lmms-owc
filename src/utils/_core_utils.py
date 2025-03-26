import datetime
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator
from itertools import islice

import pkg_resources
import pytz
from pydantic import HttpUrl, TypeAdapter
from tqdm import tqdm

from src.utils._logging_utils import get_logger

__all__ = [
    "create_iterator",
    "check_str_is_http",
    "get_datetime_str",
    "get_git_commit_hash",
    "get_progress_bar",
    "hash_string",
    "package_available",
    "parse_string_args",
    "pattern_match",
    "remove_trailing_none",
    "sanitize_list",
    "sanitize_long_string",
    "sanitize_model_name",
    "sanitize_task_name",
]

log = get_logger(__name__, rank_zero_only=True)

BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"


def check_str_is_http(x: str) -> str:
    """Check if a string is a valid HTTP URL.

    Args:
    ----
        x (str): The string to check.

    """
    http_url_adapter = TypeAdapter(HttpUrl)
    return str(http_url_adapter.validate_python(x))


def create_iterator(
    raw_iterator: Iterator,
    rank: int | None,
    world_size: int | None,
    limit: int | None = None,
) -> Iterator:
    """Create an iterator from a raw document iterator.

    Args:
    ----
        raw_iterator (Iterable): The raw document iterator.
        rank (int, optional): The rank of the iterator.
        world_size (int, optional): The size of the world.
        limit (int, optional): The limit of the iterator. Defaults to None.

    """
    return islice(raw_iterator, rank, limit, world_size)


def get_datetime_str(timezone: str = "Asia/Singapore") -> str:
    """Get the current datetime in the specified timezone as a string.

    Args:
    ----
        timezone (str, optional): The timezone name as per IANA timezone database.
            Defaults to "Asia/Singapore" (UTC+8).

    """
    tz = pytz.timezone(timezone)
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    local_time = utc_now.astimezone(tz)
    return local_time.strftime("%Y%m%d_%H%M%S")


def get_git_commit_hash() -> str | None:
    """Retrieve the current git commit hash of the repository."""
    try:
        git_path = shutil.which("git")

        if not git_path:
            raise FileNotFoundError("Git executable not found")

        result = subprocess.run(  # noqa: S603
            [git_path, "describe", "--always"],
            capture_output=True,
            shell=False,  # noqa: S603
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        git_hash = result.stdout.strip()
    except subprocess.CalledProcessError:
        git_hash = None
    except FileNotFoundError:
        git_hash = None
    return git_hash


def get_progress_bar(iterable: Iterable | None = None, **kwargs) -> tqdm:
    """Return a progress bar.

    Args:
    ----
        iterable (Iterable, optional): The iterable to wrap with a progress bar.
        kwargs: Keyword arguments to pass to the progress bar.

    """
    position = 2 * kwargs.pop("position") if "position" in kwargs else None

    return tqdm(
        iterable=iterable,
        desc=kwargs.pop("desc", "Processing"),
        position=position,
        disable=kwargs.pop("disable", False),
        leave=kwargs.pop("leave", True),
        dynamic_ncols=kwargs.pop("dynamic_ncols", True),
        file=kwargs.pop("file", sys.stdout),
        smoothing=kwargs.pop("smoothing", 0),
        bar_format=kwargs.pop("bar_format", BAR_FORMAT),
        **kwargs,
    )


def hash_string(string: str) -> str:
    """Hashes a string using SHA-256.

    Args:
    ----
        string (str): The string to hash.

    """
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    Args:
    ----
        package_name (str): Name of the package to check.

    """
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


def _string_arg_to_type(arg: str) -> bool | int | float | str:
    """Convert a string argument to a Python object.

    Args:
    ----
        arg (str): The string argument to convert

    """
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def parse_string_args(args_string: str) -> dict:
    """Parse a string of arguments into a dictionary.

    Args:
    ----
        args_string (str): The string of arguments to parse.

    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {k: _string_arg_to_type(v) for k, v in [arg.split("=") for arg in arg_list]}
    return args_dict


def pattern_match(patterns: str | list[str], source_list: list[str]) -> list[str]:
    """Match patterns to a source list and return the matched values.

    Args:
    ----
        patterns (str | list[str]): The patterns to match.
        source_list (list[str]): The list to match the patterns against.

    """
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        try:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        except KeyError as e:
            log.error("Error matching pattern %s: %s", pattern, e)

    return sorted(list(task_names))


def remove_trailing_none(string: str) -> str:
    """Remove trailing ',none' substring from the string.

    Args:
    ----
        string (str): The string to be processed.

    """
    pattern = re.compile(r",none$")
    result = re.sub(pattern, "", string)

    return result


def sanitize_list(li: list) -> list:
    """Sanitize a list by converting all inner components of a list to strings.

    Args:
    ----
        li (list): The list to be sanitized.

    """
    if isinstance(li, list):
        return [sanitize_list(item) for item in li]
    elif isinstance(li, tuple):
        return tuple(sanitize_list(item) for item in li)

    return str(li)


def sanitize_long_string(s: str, max_length: int = 40) -> str:
    """Truncate and sanitize a long string to a specified maximum length.

    Args:
    ----
        s (str): The input string to be sanitized.
        max_length (int, optional): Maximum allowed length of the output string.
            Defaults to 40 characters.

    """
    if len(s) > max_length:
        return s[: max_length // 2] + "..." + s[-max_length // 2 :]
    return s


def sanitize_model_name(model_name: str, full_path: bool = False) -> str:
    """Sanitize a model name.

    Args:
    ----
        model_name (str): The model name to be sanitized.
        full_path (bool): Whether to return the full path or not. Defaults to False.

    """
    if full_path:
        return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)

    # For models that are in Hugging Face Hub, e.g., lmms-lab/llava-onevision-qwen2-0.5b
    parts = model_name.split("/")
    last_two = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", last_two)


def sanitize_task_name(task_name: str) -> str:
    """Sanitize a task name.

    Args:
    ----
        task_name (str): The task name to be sanitized.

    """
    return re.sub(r"\W", "_", task_name)
