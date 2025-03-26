import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()


from src import utils  # noqa: E402
from src.data.metrics import get_metric_info  # noqa: E402

log = utils.get_logger(__name__, rank_zero_only=True)


def main(args: Namespace) -> None:
    """Evaluate metrics on models' outputs.

    Args:
    ----
        args (argparse.Namespace): The console arguments passed to the script.

    """
    log.setLevel(args.log_level)

    if args.seed:
        log.info("Setting random seed to %s", args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Resolve globs on the args.input
    input_paths = sorted(Path().glob(args.input)) if "*" in args.input else [Path(args.input)]

    # Find all the *_samples_*.jsonl files
    input_files_per_path = [
        list(input_path.glob("**/*_samples_*.jsonl")) if input_path.is_dir() else [input_path]
        for input_path in input_paths
    ]
    input_files = sorted(map(str, sum(input_files_per_path, [])))

    log.info("Found %d `jsonl` files to process", len(input_files))
    log.info("Expect all run paths in the form of `logs/schedule/{task_name}/{model_name}/")

    metrics_to_save_intermediate_values = [
        "concept_semantic_similarity",
        "mean_average_semantic_similarity",
        "semantic_similarity",
        "textual_inclusion_llama32",
    ]

    tasks_outputs = {}
    for input_file in input_files:
        log.debug("Loading input file %s...", input_file)
        task_name = Path(input_file).parent.parent.name
        model_name = Path(input_file).parent.name

        df = pd.read_json(input_file, lines=True)
        predictions = df["filtered_resps"].tolist()
        references = df["target"].tolist()

        # For multi-turn generation, we have an inner list to remove
        if isinstance(predictions[0], list) and isinstance(predictions[0][0], list):
            predictions = [prediction[0] for prediction in predictions]

        items = list(zip(references, predictions, strict=True))

        metric_outputs = {}
        metric_outputs["_num_samples"] = len(items)
        for metric_name in args.metrics.split(","):
            log.debug(
                "Evaluating %s on %s with model %s...",
                metric_name.replace("_", " "),
                task_name,
                model_name,
            )
            metric_info = get_metric_info(metric_name)
            if metric_info.name in ["textual_inclusion"]:
                predictions = [prediction[-1] for prediction in predictions]
                output = metric_info.builder_fn(predictions, references)
            elif metric_info.name in metrics_to_save_intermediate_values:
                log.warning(
                    'Setting `reduce="none"` for %s to save intermediate values', metric_info.name
                )
                output = metric_info.builder_fn(items)
                output = metric_info.group_fn(output, reduce="none")

                extra_columns_to_save = {}
                if metric_info.name == "concept_semantic_similarity":
                    # The metric returns a list of tuple with the concepts and the similarities
                    # and we want to save both
                    concepts = list(zip(*output, strict=True))[0]
                    similarities = list(zip(*output, strict=True))[1]

                    # Update the outputs to make it compatible with the rest
                    output = [np.max(row) for row in similarities]

                    # Save the concepts and similarities
                    extra_columns_to_save["last_resp_concepts"] = concepts
                    extra_columns_to_save["last_resp_concepts_similarities"] = similarities
                elif metric_info.name == "mean_average_semantic_similarity":
                    # The metric returns a dict of results at different thresholds
                    # and we want to save all but the semantic_similarity@avg as it is
                    # uninformative on a per-sample basis.
                    mean_average_semantic_similarity = output.pop("semantic_similarity@avg")
                    extra_columns_to_save.update(output)

                    output = mean_average_semantic_similarity

                log.info(
                    "Saving intermediate values of %s in %s",
                    metric_info.name,
                    input_file,
                )
                df[metric_info.name] = output
                for key in extra_columns_to_save:
                    df[key] = extra_columns_to_save[key]

                df.to_json(input_file, lines=True, orient="records")

                output = np.mean(output)
            else:
                output = metric_info.builder_fn(items)
                output = metric_info.group_fn(output)

            if isinstance(output, dict):
                metric_outputs.update(output)
            else:
                metric_outputs[metric_name] = output

        if task_name not in tasks_outputs:
            tasks_outputs[task_name] = {}

        if model_name not in tasks_outputs[task_name]:
            tasks_outputs[task_name][model_name] = metric_outputs
        else:
            prev_task_length = tasks_outputs[task_name][model_name]["_num_samples"]
            curr_task_length = metric_outputs["_num_samples"]
            log.warning(
                "Found multiple runs with `task_name=%s` and `model_name=%s`."
                " The previous has %d samples, the current has %d."
                " Keeping the one with more samples (or oldest if even).",
                task_name,
                model_name,
                prev_task_length,
                curr_task_length,
            )
            if curr_task_length > prev_task_length:
                tasks_outputs[task_name][model_name] = metric_outputs

    for task_name in tasks_outputs:
        task_outputs = tasks_outputs[task_name]

        metrics_names = [list(task_outputs[model_name].keys()) for model_name in task_outputs]
        metrics_names = sum(metrics_names, [])  # Flatten list
        metrics_names = sorted(list(set(metrics_names)))  # Remove duplicates and sort

        for metric_name in metrics_names:
            if metric_name.startswith("_"):  # Skip metadata
                continue

            task_metric_repr = ""
            task_metric_repr += f"{metric_name.capitalize().replace('_', ' ')} on {task_name}:\n"
            for model_name in task_outputs:
                metric_value = task_outputs[model_name][metric_name]
                task_metric_repr += f"{model_name:<29}: {metric_value:.3f}\n"
            print(task_metric_repr)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the folder/file containing the samples to process",
    )
    parser.add_argument(
        "-m",
        "--metrics",
        required=True,
        type=str,
        help="List of comma-separated metrics to evaluate on the data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility (default: 1234)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    main(args)
