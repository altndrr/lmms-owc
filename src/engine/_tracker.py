import json
import os
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfApi, hf_hub_url
from huggingface_hub.utils import (
    HfHubHTTPError,
    build_hf_headers,
    get_session,
    hf_raise_for_status,
)

from src import utils

__all__ = ["EngineTracker"]

log = utils.get_logger(__name__, rank_zero_only=True)


@dataclass(init=False)
class GeneralConfigTracker:
    """Tracker for the evaluation parameters.

    Args:
    ----
        model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
        model_name (str): Name of the model.
        model_name_sanitized (str): Sanitized model name for directory creation.
        start_time (float): Start time of the experiment. Logged at class init.
        end_time (float): Start time of the experiment. Logged when calling
            [`GeneralConfigTracker.log_end_time`]
        total_evaluation_time_seconds (str): Inferred total evaluation time in seconds (from the
            start and end times).

    """

    model_source: str | None = None
    model_name: str | None = None
    model_name_sanitized: str | None = None
    system_instruction: str | None = None
    system_instruction_sha: str | None = None
    fewshot_as_multiturn: bool | None = None
    chat_template: str | None = None
    chat_template_sha: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    total_evaluation_time_seconds: str | None = None

    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    @staticmethod
    def _get_model_name(model_args: str) -> str:
        """Extract the model name from the model arguments.

        Args:
        ----
            model_args (str): The model arguments.

        """

        def extract_model_name(model_args: str, key: str) -> str:
            """Extract the model name from the model arguments using a key.

            Args:
            ----
                model_args (str): The model arguments.
                key (str): The key to extract the model name from.

            """
            args_after_key = model_args.split(key)[1]
            return args_after_key.split(",")[0]

        # Order does matter, e.g., peft and delta are provided together with pretrained
        prefixes = ["peft=", "delta=", "pretrained=", "model=", "path=", "engine="]
        for prefix in prefixes:
            if prefix in model_args:
                return extract_model_name(model_args, prefix)

        return ""

    def log_experiment_args(
        self,
        model_source: str,
        model_args: str,
        system_instruction: str,
        chat_template: str,
        fewshot_as_multiturn: bool,
    ) -> None:
        """Log model parameters and job ID.

        Args:
        ----
            model_source (str): Source of the model (e.g. Hugging Face, GGUF, etc.)
            model_args (str): Arguments used to load the model.
            system_instruction (str): System instruction used for the evaluation.
            chat_template (str): Chat template used for the evaluation.
            fewshot_as_multiturn (bool): Whether the fewshot is used as multi-turn.

        """
        self.model_source = model_source
        self.model_name = GeneralConfigTracker._get_model_name(model_args)
        self.model_name_sanitized = utils.sanitize_model_name(self.model_name)
        self.system_instruction = system_instruction
        self.system_instruction_sha = (
            utils.hash_string(system_instruction) if system_instruction else None
        )
        self.chat_template = chat_template
        self.chat_template_sha = utils.hash_string(chat_template) if chat_template else None
        self.fewshot_as_multiturn = fewshot_as_multiturn

    def log_end_time(self) -> None:
        """Log the end time of the evaluation and calculates the total evaluation time."""
        self.end_time = time.perf_counter()
        self.total_evaluation_time_seconds = str(self.end_time - self.start_time)


class EngineTracker:
    """Track and save relevant information of the engine process.

    It compiles the data from trackers and writes it to files, which can be published to the
    HuggingFace hub if requested.

    Args:
    ----
        output_path (str, optional): Path to save the results. If not provided, the results won't
            be saved. Defaults to None.
        hub_results_org (str): The Hugging Face organization to push the results to. If not
            provided, the results will be pushed to the owner of the Hugging Face token. Defaults
            to "".
        hub_repo_name (str): The name of the Hugging Face repository to push the results to. If not
            provided, the results will be pushed to `lm-eval-results`. Defaults to "".
        details_repo_name (str): The name of the Hugging Face repository to push the details to. If
            not provided, the results will be pushed to `lm-eval-results`. Defaults to "".
        results_repo_name (str): The name of the Hugging Face repository to push the results to. If
            not provided, the results will not be pushed and will be found in the details_hub_repo.
            Defaults to "".
        push_results_to_hub (bool): Whether to push the results to the Hugging Face hub. Defaults
            to False.
        push_samples_to_hub (bool): Whether to push the samples to the Hugging Face hub. Defaults
            to False.
        public_repo (bool): Whether to push the results to a public or private repository. Defaults
            to False.
        token (str): Token to use when pushing to the Hugging Face hub. This token should have
            write access to `hub_results_org`. Defaults to "".
        leaderboard_url (str): URL to the leaderboard on the Hugging Face hub on the dataset card.
            Defaults to "".
        point_of_contact (str): Contact information on the Hugging Face hub dataset card. Defaults
            to "".
        gated (bool): Whether to gate the repository. Defaults to False.

    """

    def __init__(
        self,
        output_path: str | None = None,
        hub_results_org: str = "",
        hub_repo_name: str = "",
        details_repo_name: str = "",
        results_repo_name: str = "",
        push_results_to_hub: bool = False,
        push_samples_to_hub: bool = False,
        public_repo: bool = False,
        token: str | None = None,
        leaderboard_url: str = "",
        point_of_contact: str = "",
        gated: bool = False,
    ) -> None:
        self.general_config_tracker = GeneralConfigTracker()

        self.output_path = output_path
        self.push_results_to_hub = push_results_to_hub
        self.push_samples_to_hub = push_samples_to_hub
        self.public_repo = public_repo
        self.leaderboard_url = leaderboard_url
        self.point_of_contact = point_of_contact
        self.api = HfApi(token=token) if token else None
        self.gated_repo = gated

        if not self.api and (push_results_to_hub or push_samples_to_hub):
            raise ValueError(
                "Hugging Face token is not defined, but 'push_results_to_hub' or"
                " 'push_samples_to_hub' is set to True. Please provide a valid Hugging Face token"
                " by setting the HF_TOKEN environment variable."
            )

        if self.api and hub_results_org == "" and (push_results_to_hub or push_samples_to_hub):
            hub_results_org = self.api.whoami()["name"]
            log.warning(
                "hub_results_org was not specified. Results will be pushed to '%s'.",
                hub_results_org,
            )

        if hub_repo_name == "":
            details_repo_name = (
                details_repo_name if details_repo_name != "" else "lmms-eval-results"
            )
            results_repo_name = results_repo_name if results_repo_name != "" else details_repo_name
        else:
            details_repo_name = hub_repo_name
            results_repo_name = hub_repo_name
            log.warning(
                "hub_repo_name was specified. Both details and results will be pushed to the same"
                " repository. Using hub_repo_name is no longer recommended, details_repo_name and"
                " results_repo_name should be used instead."
            )

        self.details_repo = f"{hub_results_org}/{details_repo_name}"
        self.details_repo_private = f"{hub_results_org}/{details_repo_name}-private"
        self.results_repo = f"{hub_results_org}/{results_repo_name}"
        self.results_repo_private = f"{hub_results_org}/{results_repo_name}-private"

    def save_results_aggregated(self, results: dict, samples: dict, datetime_str: str) -> None:
        """Save the aggregated results and samples.

        Args:
        ----
            results (dict): The aggregated results to save.
            samples (dict): The samples results to save.
            datetime_str (str): The datetime string to use for the results file.

        """
        self.general_config_tracker.log_end_time()

        if self.output_path:
            try:
                log.info("Saving results aggregated")

                # Calculate cumulative hash for each task - only if samples are provided
                task_hashes = {}
                if samples:
                    for task_name, task_samples in samples.items():
                        sample_hashes = [
                            s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                            for s in task_samples
                        ]
                        task_hashes[task_name] = utils.hash_string("".join(sample_hashes))

                # Update initial results dict
                results.update({"task_hashes": task_hashes})
                results.update(asdict(self.general_config_tracker))
                dumped = json.dumps(
                    results,
                    indent=2,
                    default=utils.convert_non_serializable,
                    ensure_ascii=False,
                )

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                self.date_id = datetime_str.replace(":", "-")
                file_results_aggregated = path.joinpath(f"{self.date_id}_results.json")
                file_results_aggregated.open("w", encoding="utf-8").write(dumped)

                if self.api and self.push_results_to_hub:
                    repo_id = self.results_repo if self.public_repo else self.results_repo_private
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    self.api.upload_file(
                        repo_id=repo_id,
                        path_or_fileobj=str(path.joinpath(f"{self.date_id}_results.json")),
                        path_in_repo=os.path.join(
                            self.general_config_tracker.model_name,
                            f"{self.date_id}_results.json",
                        ),
                        repo_type="dataset",
                        commit_message=(
                            "Adding aggregated results for "
                            + self.general_config_tracker.model_name
                        ),
                    )
                    log.info(
                        "Successfully pushed aggregated results to the Hugging Face Hub. You can"
                        " find them at: %s",
                        repo_id,
                    )

            except (OSError, json.JSONDecodeError, HfHubHTTPError, ValueError) as e:
                log.warning("Could not save results aggregated")
                log.info(repr(e))
        else:
            log.info("Output path not provided, skipping saving results aggregated")

    def save_results_samples(self, task_name: str, samples: dict) -> None:
        """Save the samples results.

        Args:
        ----
            task_name (str): The task name to save the samples for.
            samples (dict): The samples results to save.

        """
        if self.output_path:
            try:
                log.info("Saving per-sample results for: %s", task_name)

                path = Path(self.output_path if self.output_path else Path.cwd())
                path = path.joinpath(self.general_config_tracker.model_name_sanitized)
                path.mkdir(parents=True, exist_ok=True)

                file_results_samples = path.joinpath(f"{self.date_id}_samples_{task_name}.jsonl")

                for sample in samples:
                    # We first need to sanitize arguments and response otherwise we won't be able
                    # to load the dataset using the datasets library
                    arguments = {}
                    for key, value in enumerate(
                        sample["arguments"][1]
                    ):  # update metadata into args
                        arguments[key] = value

                    sample["input"] = sample["arguments"][0]
                    sample["resps"] = utils.sanitize_list(sample["resps"])
                    sample["filtered_resps"] = utils.sanitize_list(sample["filtered_resps"])
                    sample["arguments"] = arguments
                    sample["target"] = str(sample["target"])

                    sample_dump = (
                        json.dumps(
                            sample,
                            default=utils.convert_non_serializable,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    with open(file_results_samples, "a", encoding="utf-8") as f:
                        f.write(sample_dump)

                if self.api and self.push_samples_to_hub:
                    repo_id = self.details_repo if self.public_repo else self.details_repo_private
                    self.api.create_repo(
                        repo_id=repo_id,
                        repo_type="dataset",
                        private=not self.public_repo,
                        exist_ok=True,
                    )
                    try:
                        if self.gated_repo:
                            headers = build_hf_headers()
                            r = get_session().put(
                                url=f"https://huggingface.co/api/datasets/{repo_id}/settings",
                                headers=headers,
                                json={"gated": "auto"},
                            )
                            hf_raise_for_status(r)
                    except Exception as e:
                        log.warning("Could not gate the repository")
                        log.info(repr(e))
                    self.api.upload_folder(
                        repo_id=repo_id,
                        folder_path=str(path),
                        path_in_repo=self.general_config_tracker.model_name_sanitized,
                        repo_type="dataset",
                        commit_message=(
                            f"Adding samples results for {task_name} to "
                            + self.general_config_tracker.model_name
                        ),
                    )
                    log.info(
                        "Successfully pushed sample results for task: %s to the Hugging Face Hub."
                        " You can find them at: %s",
                        task_name,
                        repo_id,
                    )

            except Exception as e:
                log.warning("Could not save sample results")
                log.info(repr(e))
        else:
            log.info("Output path not provided, skipping saving sample results")

    def recreate_metadata_card(self) -> None:
        """Create a metadata card for the evaluation results dataset."""
        log.info("Recreating metadata card")
        repo_id = self.details_repo if self.public_repo else self.details_repo_private

        files_in_repo = self.api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        results_files = utils.get_results_filenames(files_in_repo)
        sample_files = utils.get_sample_results_filenames(files_in_repo)

        # Build a dictionary to store the latest evaluation datetime for:
        # - each tested model and its aggregated results
        # - each task and sample results, if existing
        # i.e., {
        #     "org__model_name__results": "2021-09-01T12:00:00"
        # }
        latest_task_results_datetime = defaultdict(lambda: datetime.min.isoformat())

        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = utils.get_task_name_from_filename(filename)
            results_datetime = utils.get_task_datetime_from_filename(filename)
            task_name_sanitized = utils.sanitize_task_name(task_name)
            # Results and sample results for the same model and task will have the same datetime
            samples_key = f"{model_name}__{task_name_sanitized}"
            results_key = f"{model_name}__results"
            latest_datetime = max(
                latest_task_results_datetime[samples_key],
                results_datetime,
            )
            latest_task_results_datetime[samples_key] = latest_datetime
            latest_task_results_datetime[results_key] = max(
                latest_task_results_datetime[results_key],
                latest_datetime,
            )

        # Create metadata card
        card_metadata = MetadataConfigs()

        # Add the latest aggregated results to the metadata card for easy access
        for file_path in results_files:
            file_path = Path(file_path)
            results_filename = file_path.name
            model_name = file_path.parent
            eval_date = utils.get_task_datetime_from_filename(results_filename)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(results_filename).name
            config_name = f"{model_name}__results"
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", latest_task_results_datetime[config_name]
            )

            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all results files are listed in the metadata card
                current_results = card_metadata.get(config_name, {"data_files": []})
                current_results["data_files"].append(
                    {"split": eval_date_sanitized, "path": [str(results_filename)]}
                )
                card_metadata[config_name] = current_results
                # If the results file is the newest, update the "latest" field in the metadata card
                card_metadata[config_name]["data_files"].append(
                    {"split": "latest", "path": [str(results_filename)]}
                )

        # Add the tasks details configs
        for file_path in sample_files:
            file_path = Path(file_path)
            filename = file_path.name
            model_name = file_path.parent
            task_name = utils.get_task_name_from_filename(filename)
            eval_date = utils.get_task_datetime_from_filename(filename)
            task_name_sanitized = utils.sanitize_task_name(task_name)
            eval_date_sanitized = re.sub(r"[^\w\.]", "_", eval_date)
            results_filename = Path("**") / Path(filename).name
            config_name = f"{model_name}__{task_name_sanitized}"
            sanitized_last_eval_date_results = re.sub(
                r"[^\w\.]", "_", latest_task_results_datetime[config_name]
            )
            if eval_date_sanitized == sanitized_last_eval_date_results:
                # Ensure that all sample results files are listed in the metadata card
                current_details_for_task = card_metadata.get(config_name, {"data_files": []})
                current_details_for_task["data_files"].append(
                    {"split": eval_date_sanitized, "path": [str(results_filename)]}
                )
                card_metadata[config_name] = current_details_for_task
                # If the samples results file is the newest, update the "latest" field in the
                # metadata card
                card_metadata[config_name]["data_files"].append(
                    {"split": "latest", "path": [str(results_filename)]}
                )

        # Get latest results and extract info to update metadata card examples
        latest_datetime = max(latest_task_results_datetime.values())
        latest_model_name = max(
            latest_task_results_datetime, key=lambda k: latest_task_results_datetime[k]
        )
        last_results_file = [f for f in results_files if latest_datetime.replace(":", "-") in f][0]
        last_results_file_path = hf_hub_url(
            repo_id=repo_id, filename=last_results_file, repo_type="dataset"
        )
        latest_results_file = load_dataset(
            "json", data_files=last_results_file_path, split="train"
        )
        results_dict = latest_results_file["results"][0]
        new_dictionary = {"all": results_dict}
        new_dictionary.update(results_dict)
        results_string = json.dumps(new_dictionary, indent=4)

        dataset_summary = "Dataset automatically created during the evaluation run of model "
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += f"[{self.general_config_tracker.model_name}](https://huggingface.co/{self.general_config_tracker.model_name})\n"  # noqa: E501
        else:
            dataset_summary += f"{self.general_config_tracker.model_name}\n"
        dataset_summary += (
            f"The dataset is composed of {len(card_metadata)-1} configuration(s), each one"
            " corresponding to one of the evaluated task.\n\nThe dataset has been created from"
            f" {len(results_files)} run(s). Each run can be found as a specific split in each"
            ' configuration, the split being named using the timestamp of the run.The "train"'
            " split is always pointing to the latest results.\n\nAn additional configuration"
            ' "results" store all the aggregated results of the run.\n\nTo load the details from a'
            " run, you can for instance do the following:\n"
        )
        if self.general_config_tracker.model_source == "hf":
            dataset_summary += (
                "```python\nfrom datasets import load_dataset\n"
                f'data = load_dataset(\n\t"{repo_id}",\n\tname="{latest_model_name}",\n'
                '\tsplit="latest"\n)\n```\n\n'
            )
        dataset_summary += (
            "## Latest results\n\n"
            f'These are the [latest results from run {latest_datetime}]({last_results_file_path.replace("/resolve/", "/blob/")})'  # noqa: E501
            " (note that there might be results for other tasks in the repos if successive evals"
            " didn't cover the same tasks. You find each in the results and the \"latest\" split"
            " for each eval):\n\n"
            f"```python\n{results_string}\n```"
        )
        card_data = DatasetCardData(
            dataset_summary=dataset_summary,
            repo_url=f"https://huggingface.co/{self.general_config_tracker.model_name}",
            pretty_name=f"Evaluation run of {self.general_config_tracker.model_name}",
            leaderboard_url=self.leaderboard_url,
            point_of_contact=self.point_of_contact,
        )
        card_metadata.to_dataset_card_data(card_data)
        card = DatasetCard.from_template(
            card_data,
            pretty_name=card_data.pretty_name,
        )
        card.push_to_hub(repo_id, repo_type="dataset")
