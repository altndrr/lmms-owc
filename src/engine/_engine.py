import json
import math
import random
from argparse import Namespace
from collections import defaultdict
from itertools import chain, islice

import numpy as np
import torch
import torch.distributed as dist

from src import utils
from src.data.tasks import (
    Task,
    TaskManager,
    get_consolidated_group_results,
    get_consolidated_results,
    get_subtasks_as_dict,
    get_tasks_as_dict,
    get_tasks_as_list,
    prepare_print_tasks,
)
from src.engine._tracker import EngineTracker
from src.models import Model, get_model

__all__ = ["evaluate", "simple_evaluate"]

log = utils.get_logger(__name__, rank_zero_only=True)


@utils.deprecated_positional
def evaluate(
    model: Model,
    task_dict: dict,
    limit: int | float | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: int | None = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: str | None = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    cli_args: Namespace | None = None,
) -> dict | None:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
    ----
        model (Model): The multi-modal model to evaluate.
        task_dict (dict): Dictionary of task names and Task instances for evaluation.
        limit (int | float, optional): Limit the number of samples per task (only for testing). If
            <1, limit is a percentage of the total number of examples. Default to None.
        cache_requests (bool): Speed up evaluation by caching the building of dataset requests.
            Set to `False` for no caching. Default to False.
        rewrite_requests_cache (bool): Whether to rewrites all of the request cache. Set to `False`
            for no caching. Default to False.
        bootstrap_iters (int, optional): Number of iterations for bootstrap statistics, used when
            calculating stderr. Set to 0 for skipping all stderr calculations. Default to 1e5.
        write_out (bool): Whether to write out an example document and model input to check task
            integrity. Default to False.
        log_samples (bool): Whether to write out all model outputs and documents for per-sample
            measurements and post-hoc analysis. Default to True.
        system_instruction (str, optional): System instruction to apply to the prompt. Default to
            None.
        apply_chat_template (bool): Whether to apply the chat template to the prompt. Default to
            False.
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as multi-turn
            conversation or a single user turn. Default to False.
        cli_args (argparse.Namespace, optional): List of arguments from the cli. Default to None

    """
    results = defaultdict(dict)
    versions = defaultdict(dict)  # Tracks each task's version
    configs = defaultdict(dict)  # Tracks the YAML configs of all chosen tasks
    samples = defaultdict(list)  # Logs info about each document evaluated
    requests = defaultdict(list)  # Tracks all Instances/requests a model must generate output on
    results_agg = defaultdict(dict)
    padding_requests = defaultdict(int)
    task_hierarchy = defaultdict(list)
    task_group_alias = defaultdict(dict)
    num_fewshot = defaultdict(int)

    RANK = model.rank
    WORLD_SIZE = model.world_size

    # Get lists of group hierarchy and each type of request
    eval_tasks = get_tasks_as_list(task_dict)
    name_to_task = {}
    if not log_samples and not all(
        "bypass" not in getattr(eval_task.task, "_metric_fn_list", {}) for eval_task in eval_tasks
    ):
        raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    for task_output in eval_tasks:
        task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args

        name_to_task[task_name] = task

        group_name = None
        task_hierarchy[task_name] = []
        if isinstance(task, tuple):
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        n_shot = configs[task_name].get("num_fewshot", 0)
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        has_group_alias = "group_alias" in configs[task_name]
        if has_group_alias and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        if limit is not None:
            limit = int(math.ceil(len(task.eval_docs) * limit)) if limit < 1.0 else int(limit)

        task.build_all_requests(
            limit=limit,
            rank=RANK,
            world_size=WORLD_SIZE,
            cache_requests=cache_requests,  # Will add them later
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=model.apply_chat_template if apply_chat_template else None,
            tokenizer_name=getattr(model, "tokenizer_name", "") if apply_chat_template else "",
        )
        log.debug(
            "Task: %s; number of requests on this rank: %i",
            task_output.task_name,
            len(task._instances),
        )
        if write_out:
            for inst in task.instances:
                if inst.doc_id < 1:
                    log.info(
                        "Task: %s; document %s; context prompt (starting on next line):\n"
                        "%s\n"
                        "(end of prompt on previous line)\n"
                        "target string or answer choice index (starting on next line):\n"
                        "%s\n"
                        "(end of target on previous line)",
                        task,
                        inst.doc_id,
                        inst.args[0],
                        task.doc_to_target(inst.doc),
                    )
                    log.info("Request: %s", str(inst))

        # Aggregate Instances by model method requested to get output.
        for instance in task.instances:
            req_type = instance.request_type
            requests[req_type].append(instance)

        if WORLD_SIZE > 1:
            instances_rnk = torch.tensor(len(task._instances), device=model.device)
            gathered_item = model.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            req_type = (
                "loglikelihood" if task.OUTPUT_TYPE == "multiple_choice" else task.OUTPUT_TYPE
            )
            # Compute number of pseudo-batches to pad with
            # ? (FSDP/DDP require even batches among ranks)
            num_pad = max(gathered_item) - gathered_item[RANK]
            # TODO may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[req_type] += num_pad

    # Execute each type of request
    for req_type, reqs in requests.items():
        log.info("Running %s requests", req_type)

        # Create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if WORLD_SIZE > 1 and padding_requests[req_type] > 0:
            for _ in range(padding_requests[req_type]):
                cloned_reqs.extend([req] * req.repeats)

        # Run requests through model
        resps = getattr(model, req_type)(cloned_reqs)

        # Put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs, strict=True):
            req.resps.append(x)

        if WORLD_SIZE > 1:
            model.accelerator.wait_for_everyone()

    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        # Collect values of metrics on all datapoints
        # Unpack results and sort back in order and return control to Task
        # TODO make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)

        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)

        # Iterate over the different filters used
        for filter_key in task.instances[0].filtered_resps:
            if not getattr(cli_args, "process_with_media", False):
                doc_iterator = utils.create_iterator(
                    enumerate(task.eval_docs_no_media),
                    rank=RANK,
                    limit=int(limit) if limit else None,
                    world_size=WORLD_SIZE,
                )
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)

            doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
            doc_iterator_for_counting = (
                islice(range(len(task.test_docs())), RANK, limit, WORLD_SIZE)
                if task.has_test_docs()
                else islice(range(len(task.validation_docs())), RANK, limit, WORLD_SIZE)
            )
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar_kwargs = dict(total=total_docs, desc="Postprocessing", disable=(RANK != 0))
            pbar = utils.get_progress_bar(**pbar_kwargs)
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )

                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        if "image" not in key:
                            saved_doc[key] = value

                    filtered_arguments = []
                    for req in requests:
                        # Check if req.args is a list of tuples, and each item in the list is a
                        # serializable object
                        for value in req.args:
                            serializable = str | int | float | bool | list | dict | type(None)
                            if isinstance(value, serializable):
                                filtered_arguments.append(value)

                    example = {
                        "doc_id": doc_id,
                        "doc": saved_doc,
                        "target": target,
                        "arguments": filtered_arguments,
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        "doc_hash": utils.hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=utils.convert_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": utils.hash_string(requests[0].arguments[0]),
                        "target_hash": utils.hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)

                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)

            pbar.close()

    if hasattr(model, "_model"):
        del model._model
        torch.cuda.empty_cache()

    # If multi-GPU, then gather data across all ranks to rank 0
    if WORLD_SIZE > 1:
        # First gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)

                torch.distributed.gather_object(per_rank_samples, object_gather_list=full_samples)

                if isinstance(full_samples, list):
                    task_output.logged_samples = list(
                        chain.from_iterable(full_samples)  # pytype: disable=wrong-arg-types
                    )

            # Then collect metrics across all ranks
            for metrics in task_output.sample_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    task_output.sample_metrics[metrics], object_gather_list=metric_list
                )

                if isinstance(metric_list, list):
                    task_output.sample_metrics[metrics] = list(
                        chain.from_iterable(metric_list)  # pytype: disable=wrong-arg-types
                    )

        dist.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        # Aggregate results over all datapoints
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters)
        consolidated_results = get_consolidated_results(eval_tasks)
        results, samples, configs, versions, num_fewshot, higher_is_better = consolidated_results

        # Calculate group metrics
        if bool(results):
            results, versions, show_group_table, *_ = get_consolidated_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtasks_dict = get_subtasks_as_dict(task_dict)

        # Collect all higher_is_better values for metrics in the group's subtasks
        # TODO clean this up
        # TODO unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtasks_dict.items():
            if len(task_list) != 0:  # Subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better:
                            _higher_is_better[m] = h

                        if _higher_is_better.get(m) is not None and _higher_is_better[m] != h:
                            log.warning(
                                "`higher_is_better` values for metric %s in group %s are"
                                " inconsistent. Defaulting to None.",
                                m,
                                group,
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(group_agg.items())} if bool(group_agg) & show_group_table else {}),
            "group_subtasks": dict(reversed(subtasks_dict.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }

        if log_samples:
            results_dict["samples"] = dict(samples)
    else:
        results_dict = None

    if hasattr(model, "accelerator"):
        model.accelerator.wait_for_everyone()

    return results_dict


@utils.deprecated_positional
def simple_evaluate(
    model_name: str,
    model_args: str | None = None,
    tasks: list[str | dict | Task] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | None = None,
    use_cache: str | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: int | float | None = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    engine_tracker: EngineTracker | None = None,
    system_instruction: str | None = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: str | None = None,
    task_manager: TaskManager | None = None,
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = utils.get_datetime_str(),
    cli_args: Namespace | None = None,
) -> dict | None:
    """Instantiate and evaluate a model on a list of tasks.

    It extends `evaluate` by handling setup, configuration, and cleanup of the evaluation
    process.

    Args:
    ----
        model_name (str): The name of the model to evaluate.
        model_args (str, optional): String arguments for each model class. Ignored if `model`
            argument is a Model object.
        tasks (list[str | dict | Task]): List of task names or Task objects. Task objects will be
            taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
        num_fewshot (int, optional): Number of examples in few-shot context. Default to None.
        batch_size (int, optional): Batch size for the model forward.
        use_cache (str, optional): A path to a sqlite db file for caching model responses. Set to
            `None` for no caching. Default to None.
        cache_requests (bool): Speed up evaluation by caching the building of dataset requests.
            Set to `False` for no caching. Default to False.
        rewrite_requests_cache (bool): Whether to rewrites all of the request cache. Set to `False`
            for no caching. Default to False.
        delete_requests_cache (bool): Whether to delete all of the request cache. Set to `False`
            for no caching. Default to False.
        limit (int | float, optional): Limit the number of samples per task (only for testing). If
            <1, limit is a percentage of the total number of examples. Default to None.
        bootstrap_iters (int, optional): Number of iterations for bootstrap statistics, used when
            calculating stderr. Set to 0 for skipping all stderr calculations. Default to 1e5.
        check_integrity (bool): Whether to run the relevant part of the test suite for the tasks.
            Default to False.
        write_out (bool): Whether to write out an example document and model input to check task
            integrity. Default to False.
        log_samples (bool): Whether to write out all model outputs and documents for per-sample
            measurements and post-hoc analysis. Default to True.
        engine_tracker (EngineTracker, optional): The tracker to use to track and save relevant
            evaluation information. Default to None.
        system_instruction (str, optional): System instruction to apply to the prompt. Default to
            None.
        apply_chat_template (bool): Whether to apply the chat template to the prompt. Default to
            False.
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as multi-turn
            conversation or a single user turn. Default to False.
        gen_kwargs (str, optional): String arguments for model generation. Ignored for all tasks
            with ``output_type="loglikelihood``. Default to None.
        task_manager (TaskManager, optional): The manager to use to index and load tasks. Default
            to None.
        predict_only (bool): Whether to only generate and return model outputs without evaluating
            metrics. Default to False.
        random_seed (int): Set the Python random seed for reproducibility. Default to 0.
        numpy_random_seed (int): Set the numpy random seed for reproducibility. Default to 1234.
        torch_random_seed (int): Set the torch random seed for reproducibility. Default to 1234.
        fewshot_random_seed (int): Set the random seed for the fewshot sampler. Default to 1234.
        datetime_str (str): The current datetime as a string. Default to ``get_datetime_str()``.
        cli_args (argparse.Namespace, optional): List of arguments from the cli. Default to None

    """
    if random_seed is not None:
        log.info("Setting random seed to %d", random_seed)
        random.seed(random_seed)

    if numpy_random_seed is not None:
        log.info("Setting numpy seed to %d", numpy_random_seed)
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        log.info("Setting torch manual seed to %d", torch_random_seed)
        torch.manual_seed(torch_random_seed)

    if not tasks:
        raise ValueError("No tasks specified, or no tasks found. Please verify the task names.")

    if gen_kwargs:
        gen_kwargs = utils.parse_string_args(gen_kwargs)
        log.warning(
            "`generation_kwargs` specified through cli, these settings will be used over set"
            " parameters in yaml tasks."
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""

    if task_manager is None:
        task_manager = TaskManager(model_name=model_name)

    task_dict = get_tasks_as_dict(tasks, task_manager)

    model_kwargs = utils.parse_string_args(model_args)
    model = get_model(model_name, batch_size=batch_size, **model_kwargs)

    model.eval()
    torch.set_grad_enabled(False)

    # Recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict: dict) -> dict:
        """Adjust task configurations recursively for all tasks and subtasks.

        Args:
        ----
            task_dict (dict): A dictionary containing tasks, where keys are task names and values
                are either Task objects or nested dictionaries containing subtasks.

        """
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if isinstance(task_obj, tuple):
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                model.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type") and gen_kwargs:
                    task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if predict_only:
                    log.info(
                        "Processing %s in output-only mode. Metrics will not be calculated!",
                        task_name,
                    )
                    # Change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # Override tasks' fewshot values to the provided num_fewshot arg value except if
                # tasks have it set to 0 manually in their configs--then we should never overwrite
                # that.
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        log.info(
                            "`num_fewshot` has been set to 0 for %s in its config. Manual"
                            " configuration will be ignored.",
                            task_name,
                        )
                    else:
                        log.warning(
                            "Overwriting default num_fewshot of %s from %s to %d",
                            task_name,
                            default_num_fewshot,
                            num_fewshot,
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)

                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        utils.run_task_tests(task_list=tasks)

    if engine_tracker is not None:
        engine_tracker.general_config_tracker.log_experiment_args(
            model_source=model_name,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=model.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        model=model,
        task_dict=task_dict,
        limit=limit,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        cli_args=cli_args,
    )

    torch.set_grad_enabled(True)

    if model.rank == 0:
        # Add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            # TODO add more model info
        }

        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(model.batch_sizes.values()) if hasattr(model, "batch_sizes") else []
                ),
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = utils.get_git_commit_hash()
        results["date"] = datetime_str
        return results
    else:
        return None
