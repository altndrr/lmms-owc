import sys
from pathlib import Path

from src.utils._decorators import deprecated_positional
from src.utils._logging_utils import get_logger

__all__ = ["run_task_tests"]

log = get_logger(__name__, rank_zero_only=True)


@deprecated_positional
def _find_test_root(start_path: Path) -> Path:
    """Search upward in the directory tree for the package root containing 'tests' folder.

    Args:
    ----
        start_path (pathlib.Path): Initial directory path to start the search from.

    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(
        f"Unable to find package root within {max_layers} upwards" + f"of {start_path}"
    )


@deprecated_positional
def run_task_tests(task_list: list[str]) -> None:
    """Execute tests for the specified tasks.

    Args:
    ----
        task_list (list[str]): List of task names to run tests for.

    """
    import pytest

    package_root = _find_test_root(start_path=Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(
            "Not all tests for the specified tasks ({task_list}) ran successfully! Error"
            f" code: {pytest_return_val}"
        )
