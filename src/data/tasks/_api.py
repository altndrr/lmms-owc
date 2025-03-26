from collections import defaultdict

import numpy as np

from src.data.tasks._base import Task, TaskOutput
from src.data.tasks._manager import ConfigurableGroup, TaskManager

__all__ = [
    "get_consolidated_group_results",
    "get_consolidated_results",
    "get_subtasks_as_dict",
    "get_tasks_as_dict",
    "get_tasks_as_list",
    "prepare_print_tasks",
]


def _aggregate_subtask_metrics(metrics: list, sizes: list, weight_by_size: bool = True) -> float:
    """Aggregate subtask metrics.

    Args:
    ----
        metrics (list): List of metrics.
        sizes (list): List of sizes.
        weight_by_size (bool, optional): Whether to weight by size.

    """
    if not weight_by_size:
        sizes = [1] * len(sizes)

    if len(metrics) != len(sizes):
        raise ValueError("Metrics and sizes must be the same length.")

    return sum([m * s for m, s in zip(metrics, sizes, strict=True)]) / sum(sizes)


def _pooled_sample_stderr(std_errs: list[float], sizes: list[int]) -> float:
    """Compute the bootstrapped standard error across subtasks in a group.

    Args:
    ----
        std_errs (list[float]): List of standard errors.
        sizes (list[int]): List of sizes.

    """
    if len(std_errs) != len(sizes):
        raise ValueError("Metrics and sizes must be the same length.")

    # Formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # This empirically seems to match running `get_metric_stderr_builder` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, std_errs, strict=True)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def get_consolidated_group_results(
    results: dict,
    versions: dict,
    task_dict: dict,
    task_root: str | None = None,
    show_group_table: bool = False,
    task_aggregation_list: dict | None = None,
) -> tuple[dict, dict, bool, dict | None]:
    """Recursively calculate aggregated metrics for groups and update results and versions.

    Args:
    ----
        results (dict[str, dict[str, Any]]): Dictionary with task names as keys and dictionaries
            with metric data as values. Each metric dictionary contains "alias" and metric/filter
            name pairs.
        versions (dict[str, float | None]): Dictionary mapping task names to version numbers (if
            specified).
        task_dict (dict): Dictionary containing tasks and groups to process metrics for.
        task_root (str | None, optional): The root task or group name. Defaults to None.
        show_group_table (bool, optional): Whether to display a group table. Defaults to False.
        task_aggregation_list (dict[str, list[str]] | None, optional): Dictionary listing subtasks
            for aggregating group metrics.  Defaults to None.

    """
    if task_root is None:
        task_root = {}

    if task_aggregation_list is None:
        task_aggregation_list = {}

    for group_or_task, group_or_task_info in task_dict.items():
        # Convert to string
        if isinstance(group_or_task, ConfigurableGroup):
            group_config = group_or_task.config
            group_or_task = group_or_task.group_name
        else:
            group_config = None

        if isinstance(group_or_task_info, Task):
            if task_root:
                task_aggregation_list.setdefault(task_root, []).append(
                    group_or_task_info.task_name
                )
        else:
            grouped_results = get_consolidated_group_results(
                results,
                versions,
                group_or_task_info,
                group_or_task,
                show_group_table,
                task_aggregation_list,
            )
            (results, versions, show_group_table, _task_aggregation_list) = grouped_results

            if task_root:
                task_aggregation_list.setdefault(task_root, []).extend(
                    task_aggregation_list.get(group_or_task, [])
                )

            if (group_config is None) or (group_config["aggregate_metric_list"] is None):
                results[group_or_task][" "] = " "
                continue

            if "aggregate_metric_list" in group_config:
                agg_metric_list = group_config["aggregate_metric_list"]

            show_group_table = show_group_table | bool(group_config["aggregate_metric_list"])

            task_list = _task_aggregation_list[group_or_task]

            metric_list = list(
                {
                    key
                    for task in task_list
                    for key in results[task]
                    if "_stderr" not in key and key not in ["task", "alias", "samples"]
                }
            )

            for metric in metric_list:
                stderr = "_stderr,".join(metric.split(","))

                # Gather metrics, sizes, and stderrs from subtasks
                # TODO copy metrics?
                metrics = [results[task][metric] for task in task_list if metric in results[task]]
                stderrs = [results[task][stderr] for task in task_list if stderr in results[task]]
                sizes = [results[task]["samples"] for task in task_list if metric in results[task]]

                for metric_config in agg_metric_list:
                    for _ in metric_config["filter_list"]:
                        if metric_config["metric"] not in metric:
                            continue

                        # Compute group's pooled metric and stderr
                        if metric_config["aggregation"] == "mean":
                            aggregate_fn = _aggregate_subtask_metrics
                        elif callable(metric_config["aggregation"]):
                            aggregate_fn = metric_config["aggregation"]
                        else:
                            raise ValueError(
                                "Currently, only 'mean' is supported for automatically"
                                " aggregating scores across groups' subtasks. Got"
                                f" '{metric_config['aggregation']}' for group '{group_or_task}'"
                            )

                        results[group_or_task][metric] = aggregate_fn(
                            metrics, sizes, metric_config["weight_by_size"]
                        )

                        # TODO calculate groups' metrics using arbitrary agg fns
                        if "N/A" in stderrs:
                            results[group_or_task][stderr] = "N/A"
                        else:
                            # NOTE this assumes we are using the mean to aggregate. There are
                            # warnings about this elsewhere
                            results[group_or_task][stderr] = _pooled_sample_stderr(stderrs, sizes)

                results[group_or_task]["samples"] = sum(sizes)
                group_metadata = group_config.get("metadata", None)
                if group_metadata is not None:
                    versions[group_or_task] = group_metadata.get("version", None)

    return results, versions, show_group_table, task_aggregation_list


def get_consolidated_results(
    eval_tasks: list[TaskOutput],
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Consolidate results from multiple evaluation tasks.

    Each evaluation task's results are consolidated into these dictionaries. The results dict
    includes metric values, stderr values, and task aliases from the task config.

    Args:
    ----
        eval_tasks (list[TaskOutput]): List of TaskOutput instances containing evaluation results.

    """
    results = defaultdict(dict)
    samples = defaultdict(list)  # Logs info about each document evaluated.
    num_fewshot = defaultdict(int)  # Store num-fewshot value per task
    configs = defaultdict(dict)  # Tracks the YAML configs of all chosen task
    versions = defaultdict(dict)  # Tracks each task's version.
    higher_is_better = defaultdict(dict)  # Track `higher_is_better` for each metric

    for task_output in eval_tasks:
        if "task_alias" in (task_config := task_output.task_config):
            results[task_output.task_name]["alias"] = task_config["task_alias"]
        else:
            results[task_output.task_name]["alias"] = task_output.task_name

        if group_alias := task_output.group_alias:  # noqa: SIM102
            if group_alias not in results and (group_name := task_output.group_name):
                results[group_name]["alias"] = group_alias

        num_fewshot[task_output.task_name] = task_output.n_shot
        configs[task_output.task_name] = task_output.task_config
        versions[task_output.task_name] = task_output.version
        samples[task_output.task_name] = task_output.logged_samples
        higher_is_better[task_output.task_name] = task_output.task.higher_is_better()

        for (metric, filter_key), _ in task_output.sample_metrics.items():
            metric_key = f"{metric},{filter_key}"
            results[task_output.task_name][metric_key] = task_output.agg_metrics[metric_key]
            results[task_output.task_name]["samples"] = task_output.sample_len
            results[task_output.task_name][
                f"{metric}_stderr,{filter_key}"
            ] = task_output.agg_metrics[f"{metric}_stderr,{filter_key}"]

    return results, samples, configs, versions, num_fewshot, higher_is_better


def get_subtasks_as_dict(task_dict: dict, task_root: str | None = None, depth: int = 0) -> dict:
    """Get a dict of subtasks from a task dictionary.

    Args:
    ----
        task_dict (dict): A dictionary containing tasks.
        task_root (str, optional): The root task name. Defaults to None.
        depth (int, optional): The depth of the task. Defaults to 0.

    """
    subtasks_dict = {}
    for group_obj, task_obj in task_dict.items():
        if isinstance(group_obj, ConfigurableGroup):
            group_name = group_obj.group_name
        else:
            group_name = group_obj
        if isinstance(task_obj, dict):
            _subtasks_dict = get_subtasks_as_dict(task_obj, task_root=group_name, depth=depth + 1)
            if task_root:
                subtasks_dict.setdefault((task_root, depth), []).extend(
                    [_task for (_task, _depth) in _subtasks_dict if (_depth - 1) == depth]
                )

            subtasks_dict = {**subtasks_dict, **_subtasks_dict}
        else:
            if isinstance(task_obj, ConfigurableGroup):
                group_or_task_name = task_obj.group_name
            elif isinstance(task_obj, Task):
                group_or_task_name = task_obj.task_name

            if task_root is None:
                subtasks_dict.setdefault((group_or_task_name, depth), [])
            else:
                subtasks_dict.setdefault((task_root, depth), []).append(group_or_task_name)

    if depth == 0:
        _subtasks_dict = {}
        for group_key, task_list in subtasks_dict.items():
            group_name, depth = group_key
            _subtasks_dict[group_name] = task_list
        subtasks_dict = _subtasks_dict

    return subtasks_dict


def _get_task_name_from_object(task_object: Task) -> str:
    """Get the task name from an object.

    Args:
    ----
        task_object (Task): The task object.

    """
    if hasattr(task_object, "config"):
        return task_object._config["task"]

    # ! This gives a mechanism for non-registered tasks to have a custom name when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


def _check_duplicates(task_dict: dict) -> None:
    """Check for duplicates in a task dictionary.

    Args:
    ----
        task_dict (dict): The task dictionary.

    """
    subtask_names = []
    for _, value in task_dict.items():
        subtask_names.extend(value)

    duplicate_tasks = {
        task_name for task_name in subtask_names if subtask_names.count(task_name) > 1
    }

    # Locate potentially problematic groups that seem to 'compete' for constituent subtasks.
    competing_groups = [
        group
        for group in task_dict
        if len(set(task_dict[group]).intersection(duplicate_tasks)) > 0
    ]

    if len(duplicate_tasks) > 0:
        raise ValueError(
            "Found 1 or more tasks while trying to call get_tasks_as_dict() that were members of"
            f" more than 1 called group: {list(duplicate_tasks)}. Offending groups:"
            f" {competing_groups}. Please call groups which overlap their constituent tasks in"
            " separate evaluation runs."
        )


def get_tasks_as_dict(
    task_name_list: str | list[str | dict | Task],
    task_manager: TaskManager | None = None,
) -> dict:
    """Get a dictionary of task objects from either a name, config, or Task object.

    Args:
    ----
        task_name_list (str, list): The list of task names.
        task_manager (TaskManager, optional): The task manager. If None, a new one will be created.
            Defaults to None.

    """
    task_name_from_string_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    if isinstance(task_name_list, str):
        task_name_list = [task_name_list]
    elif isinstance(task_name_list, list):
        if not all([isinstance(task, str | dict | Task) for task in task_name_list]):
            raise TypeError(
                "Expected all list items to be of types 'str', 'dict', or 'Task', but at least"
                " one entry did not match."
            )
    else:
        raise TypeError(f"Expected a 'str' or 'list' but received {type(task_name_list)}.")

    string_task_name_list = [task for task in task_name_list if isinstance(task, str)]
    others_task_name_list = [task for task in task_name_list if not isinstance(task, str)]
    if len(string_task_name_list) > 0:
        if task_manager is None:
            task_manager = TaskManager()

        task_name_from_string_dict = task_manager.load_task_or_group(string_task_name_list)

    if task_manager is None:
        raise ValueError("task_manager cannot be None.")

    for task_element in others_task_name_list:
        if isinstance(task_element, dict):
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                **task_manager.load_config(config=task_element),
            }
        elif isinstance(task_element, Task):
            task_name_from_object_dict = {
                **task_name_from_object_dict,
                _get_task_name_from_object(task_element): task_element,
            }

    task_name_from_string_dict_keys = set(task_name_from_string_dict.keys())
    if not task_name_from_string_dict_keys.isdisjoint(set(task_name_from_object_dict.keys())):
        raise ValueError("Task names from string and object are overlapping.")

    final_task_dict = {
        **task_name_from_string_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }

    # Behavior can get odd if one tries to invoke several groups that "compete" for the same task
    # (notably, because one could request several num_fewshot values at once in GroupConfig
    # overrides for the subtask and we'd be unsure which to use and report).
    # For this reason, we explicitly check for duplicates.
    _check_duplicates(get_subtasks_as_dict(final_task_dict))

    return final_task_dict


def get_tasks_as_list(task_dict: dict) -> list[TaskOutput]:
    """Recursively extract TaskOutput objects from a nested task dictionary.

    Args:
    ----
        task_dict (dict): A dictionary containing task names as keys and either TaskOutput objects
            or nested dictionaries of tasks as values.

    """
    outputs = []
    for task_name, task_obj in task_dict.items():
        if isinstance(task_obj, dict):
            _outputs = get_tasks_as_list(task_obj)
            outputs.extend(_outputs)
        else:
            task_output = TaskOutput.from_task_dict(task_name, task_obj)
            outputs.append(task_output)

    return outputs


def prepare_print_tasks(
    task_dict: dict, results: dict, task_depth: int = 0, group_depth: int = 0
) -> tuple[dict, dict]:
    """Prepare task hierarchy and aggregate results for printing.

    Processes task dictionary recursively to create hierarchical aggregations
    of task and group results with proper indentation levels.

    Args:
    ----
        task_dict (dict): Dictionary representing task group hierarchy. Each key is a
            group name with value as list of task names.
        results (dict): Dictionary containing results for each task. Each key is a
            group name with value as dictionary of task results.
        task_depth (int, optional): Indentation level for task hierarchy. Defaults to 0.
        group_depth (int, optional): Indentation level for group hierarchy. Defaults to 0.

    """

    def _sort_task_dict(task_dict: dict) -> dict:
        """Sort the task dictionary at the current hierarchy level by task name.

        Performs sorting of tasks within each sub-header alphabetically. Tasks are sorted based on
        their group name if they are ConfigurableGroup instances, otherwise by their task name.

        Args:
        ----
            task_dict (dict): Dictionary containing tasks to be sorted.

        """
        return dict(
            sorted(
                task_dict.items(),
                key=lambda item: item[0].group_name
                if isinstance(item[0], ConfigurableGroup)
                else item[0],
            )
        )

    task_agg = defaultdict(dict)
    group_agg = defaultdict(dict)
    task_dict = _sort_task_dict(task_dict)
    for task_or_group_name, task_or_group_obj in task_dict.items():
        tab_string = " " * task_depth + "- " if task_depth > 0 else ""
        if isinstance(task_or_group_name, ConfigurableGroup):
            name = task_or_group_name.group_name
            from_configurable_group = True
            task_or_group_obj = _sort_task_dict(task_or_group_obj)
        elif isinstance(task_or_group_name, str):
            name = task_or_group_name
            if isinstance(task_or_group_obj, Task):
                name = task_or_group_obj.task_name
            from_configurable_group = False

        task_agg[name] = results[name].copy()
        if from_configurable_group:
            if task_or_group_name.group_alias is not None:
                alias = task_or_group_name.group_alias
            else:
                alias = task_or_group_name.group
        else:
            alias = task_agg[name].get("alias", name)

        task_agg[name]["alias"] = tab_string + alias
        if "samples" in task_agg[name]:
            task_agg[name].pop("samples")

        if from_configurable_group and (" " not in results[name]):
            group_tab_string = " " * group_depth + "- " if group_depth > 0 else ""
            group_agg[name] = results[name].copy()
            group_agg[name]["alias"] = group_tab_string + alias
            if "samples" in group_agg[name]:
                group_agg[name].pop("samples")

        if isinstance(task_or_group_obj, dict):
            task_depth += 1
            group_depth += 1
            _task_agg, _group_agg = prepare_print_tasks(
                task_or_group_obj, results, task_depth, group_depth
            )
            task_agg = {
                **task_agg,
                **_task_agg,
            }
            group_agg = {**group_agg, **_group_agg}
            task_depth -= 1
            group_depth -= 1

    return task_agg, group_agg
