import argparse
import datetime
import json
import logging
import os
import sys
import traceback
import warnings
from functools import partial

import yaml
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from dotenv import load_dotenv

load_dotenv()

from src import utils  # noqa: E402
from src.data.loggers import WandbLogger  # noqa: E402
from src.data.tasks import TaskManager, get_tasks_as_dict  # noqa: E402
from src.engine import EngineTracker, simple_evaluate  # noqa: E402

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
log = utils.get_logger(__name__, rank_zero_only=True)


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
) -> list[int | None]:
    """Convert a comma-separated string of integers or 'None' values into a list.

    Args:
    ----
        min_len (int): Minimum length of the resulting list.
        max_len (int): Maximum length of the resulting list.
        defaults (str): Default values to use if input is empty.
        value (str): Input string to parse.
        split_char (str): Character to split the input string on. Default to ",".

    """

    def parse_value(item: str) -> int | None:
        """Parse a string into either an integer or None.

        Args:
        ----
            item: A string that represents either an integer number or 'none'/'None'.

        """
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None") from e

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            "Argument requires %d integers or None, separated by '%s'. "
            "Missing values will be filled with defaults.",
            max_len,
            split_char,
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # Extend items list with missing defaults

    return items


def _run_single_evaluation(args: argparse.Namespace) -> tuple[dict, dict] | tuple[None, None]:
    """Execute single-run evaluation from command line arguments.

    Args:
    ----
        args (argparse.Namespace): Parsed command line arguments.

    """
    if args.include_path is not None:
        log.info("Including path: %s", args.include_path)
    task_manager = TaskManager(include_path=args.include_path, model_name=args.model)

    # Update the engine tracker args with the output path and the HF token
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}"
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"

    engine_tracker_args = utils.parse_string_args(args.hf_hub_log_args)
    log.info("Engine tracker args: %s", engine_tracker_args)

    engine_tracker = EngineTracker(**engine_tracker_args)

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError(
            "If fewshot_as_multiturn is set, apply_chat_template must be set to True."
        )

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        log.info("Including path: %s", args.include_path)

    if "push_samples_to_hub" in engine_tracker_args and not args.log_samples:
        log.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be"
            " pushed to the Hub."
        )

    if args.limit:
        log.warning(
            "The --limit argument should only be used for testing. Real metrics should not be"
            " computed using limit."
        )

    if args.tasks is None:
        log.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        log.info(
            "Available Tasks:\n - %s",
            "\n - ".join(sorted(task_manager.list_all_tasks())),
        )
        sys.exit()
    elif args.tasks == "list_groups":
        log.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        log.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        log.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_with_num":
        log_message = (
            "\n"
            + "=" * 70
            + "\n"
            + "\n\tYou are trying to check all the numbers in each task."
            + "\n\tThis action will download the complete dataset."
            + "\n\tIf the results are not clear initially, call this again."
            + "\n\n"
            + "=" * 70
        )
        log.info(log_message)
        for task_name in sorted(task_manager.list_all_tasks()):
            try:
                task_dict = get_tasks_as_dict([task_name])
                task_obj = task_dict[task_name]
                if isinstance(task_obj, tuple):
                    _, task_obj = task_obj
                    if task_obj is None:
                        continue
                log.info(
                    "\nTask : %s\n - #num : %d",
                    task_obj.config.task,
                    len(task_obj.test_docs())
                    if task_obj.has_test_docs()
                    else len(task_obj.validation_docs()),
                )
            except Exception as e:
                log.debug("\nTask : %s fail to load \n Exception : \n %s", task_name, e)
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # We don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                log.error(
                    "Tasks were not found: %s\n"
                    "                                               Try `eval_model --tasks list`"
                    " for list of available tasks",
                    missing,
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `eval_model --tasks"
                    f" {{list_groups,list_subtasks,list_tags,list}}` to list out all available"
                    " names for task groupings; only (sub)tasks; tags; or all of the above, or"
                    " pass '--log_level DEBUG' to troubleshoot task registration issues."
                )

    log.info("Selected Tasks: %s", task_names)
    request_caching_args = {
        "cache_requests": args.cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": args.cache_requests == "refresh",
        "delete_requests_cache": args.cache_requests == "delete",
    }

    datetime_str = args.datetime_str

    results = simple_evaluate(
        model_name=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        engine_tracker=engine_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        **request_caching_args,
    )

    if isinstance(results, dict):
        samples = results.pop("samples") if args.log_samples else None
        dumped = json.dumps(results, indent=4, default=utils.convert_non_serializable)
        if args.show_config:
            print(dumped)

        engine_tracker.save_results_aggregated(
            results=results,
            samples=samples if args.log_samples else None,
            datetime_str=datetime_str,
        )

        if args.log_samples:
            for task_name, _ in results["configs"].items():
                engine_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if engine_tracker.push_results_to_hub or engine_tracker.push_samples_to_hub:
            engine_tracker.recreate_metadata_card()

        return results, samples

    return None, None


def main(args: argparse.Namespace) -> None:
    """Evaluate model on tasks.

    Args:
    ----
        args (Namespace): Parsed command line arguments.

    """
    if len(sys.argv) == 1:
        print("┌────────────────────────────────────────────────────────────────────────────────┐")
        print("│ Please provide arguments to evaluate the model. e.g.,                          │")
        print("│ `eval_model.py --model llava --tasks okvqa`                                    │")
        print("│ Use `eval_model --help` for more information.                                  │")
        print("└────────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1)

    if args.wandb_args:
        if "name" not in args.wandb_args:
            name = (
                f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            )
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**utils.parse_string_args(args.wandb_args))

    # Reset logger
    log.info("Log level set to %s", args.log_level)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config) as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if not isinstance(config_args, list) else config_args

        # Multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # Initialize Accelerator
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
    is_main_process = bool(accelerator.is_main_process)

    for args in args_list:
        try:
            # Set the same start timestamp for everybody
            args.datetime_str = accelerator.gather_for_metrics(
                [utils.get_datetime_str(timezone=args.timezone)], use_gather_object=True
            )[0]

            results, samples = _run_single_evaluation(args)
            results_list.append(results)

            accelerator.wait_for_everyone()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    log.info("Logging to Weights and Biases failed due to %s", e)

        except Exception as e:
            if args.log_level == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                log.error(
                    "Error during evaluation: %s. Please set `--log_level=DEBUG` to get more"
                    " information.",
                    e,
                )
                results_list.append(None)

    for args, results in zip(args_list, results_list, strict=True):
        # The output is None if the process is not the main process (rank 0)
        if results is not None:
            print(
                f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit:"
                f" {args.limit}, num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
            )
            print(utils.make_table(results))
            if "groups" in results:
                print(utils.make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default="",
        help=(
            "Path to a yaml file specifying all eval arguments, will ignore cli arguments if"
            " specified"
        ),
    )
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command `eval_model` --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="Batch size for the model forward. Default 1.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help=(
            "The path to the output file where the result metrics will be saved. If the path is a"
            " directory and log_samples is true, the results will be saved in the directory. Else"
            " the parent directory will be used."
        ),
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help=(
            "Speed up evaluation by caching the building of dataset requests. `None` if not"
            " caching."
        ),
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help=(
            "If True, write out all model outputs and documents for per-sample measurement and"
            " post-hoc analysis"
        ),
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help=(
            "If True, write out all model outputs and documents for per-sample measurement and"
            " post-hoc analysis to Weights and Biases"
        ),
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`"
        ),
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help=(
            "Comma separated string arguments passed to wandb.init, e.g.,"
            "`project=eval_model,job_type=eval"
        ),
    )
    parser.add_argument(
        "--timezone",
        default="Europe/Rome",
        help=(
            "Timezone for datetime string, e.g. Asia/Singapore, America/New_York. You can check"
            " the full list via `import pytz; print(pytz.common_timezones)`"
        ),
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help=(
            "Comma separated string arguments passed to Hugging Face Hub's log function, e.g.,"
            " `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`"
        ),
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help=(
            "Use with --log_samples. Only model outputs will be saved and metrics will not be"
            " evaluated."
        ),
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and"
            " fewshot sampling seeds, respectively, or a single integer to set the same seed for"
            " all four.\n"
            "The values are either an integer or 'None' to not set the seed. Default is"
            f" `{default_seed_string}` (for backward compatibility).\n"
            "E.g., `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot"
            " sampling seed to 52. Here numpy's seed is not set since the second value is"
            " `None`.\n"
            "E.g., `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--process_with_media",
        action="store_true",
        help="Whether you will process you dataset with audio, image.",
    )
    args = parser.parse_args()

    main(args)
