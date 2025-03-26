import random
from argparse import ArgumentParser, Namespace
from collections import Counter
from itertools import combinations
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

from src import utils  # noqa: E402

log = utils.get_logger(__name__, rank_zero_only=True)


def _elo_rating(
    rating_a: int, rating_b: int, score_a: int, k_factor: int = 32, zero_sum: bool = False
) -> tuple[float, float]:
    """Calculate updated Elo ratings for two players after a match.

    Args:
    ----
        rating_a (int): Current Elo rating of player A
        rating_b (int): Current Elo rating of player B
        score_a (int): Game outcome for player A (1 for win, 0 for loss)
        k_factor (int): Factor determining rating adjustment magnitude
        zero_sum (bool): Whether to do zero-sum adjustment. Useful with small player bases and
            to avoid score inflation/deflation. Default to False.

    """
    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

    if zero_sum:
        change_a = k_factor * (score_a - expected_a)
        change_b = k_factor * ((1 - score_a) - expected_b)

        average_change = (change_a - change_b) / 2
        adjusted_change_a, adjusted_change_b = average_change, -average_change

        new_rating_a = rating_a + adjusted_change_a
        new_rating_b = rating_b + adjusted_change_b
    else:
        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)

    return new_rating_a, new_rating_b


def _sample_games(task_inputs: dict, n: int) -> list[dict]:
    """Sample n games from the complete list.

    Args:
    ----
        task_inputs (dict): The dictionary of players with their games.
        n (int): The number of samples to return.

    """
    player_names = list(task_inputs.keys())

    game_results = task_inputs[player_names[0]][["doc_id", "target"]]
    for model_name in task_inputs:
        right = task_inputs[model_name][["doc_id", "filtered_resps"]]
        right = right.rename(columns={"filtered_resps": model_name})
        game_results = pd.merge(game_results, right, how="left", on="doc_id")

    game_sampled = []
    for _ in range(n):
        idx = random.sample(range(len(game_results)), 1)[0]
        players = random.sample(list(combinations(player_names, 2)), 1)[0]

        game_result = game_results.iloc[idx][["doc_id", "target", *players]]
        game = dict(
            doc_id=game_result["doc_id"].item(),
            player_a_name=players[0],
            player_a_response=game_result[players[0]][-1],  # Keep only the last response
            player_b_name=players[1],
            player_b_response=game_result[players[1]][-1],  # Keep only the last response
            reference=game_result["target"],
        )

        game_sampled.append(game)

    return game_sampled


def main(args: Namespace) -> None:
    """Evaluate Elo ratings on models' outputs.

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

    input_path = Path(args.input)

    if input_path.is_file():
        raise ValueError("--input should be a folder containing multiple runs.")

    input_files = input_path.glob("**/*_samples_*.jsonl")
    input_files = sorted([str(input_file) for input_file in input_files])

    log.info("Expect all run paths in the form of `logs/schedule/{task_name}/{model_name}/")

    # Load the input data
    tasks_inputs = {}
    for input_file in input_files:
        log.debug("Loading input file %s...", input_file)
        task_name = Path(input_file).parent.parent.name
        model_name = Path(input_file).parent.name

        df = pd.read_json(input_file, lines=True)
        df = df[["doc_id", "filtered_resps", "target"]].sort_values("doc_id")

        if task_name not in tasks_inputs:
            tasks_inputs[task_name] = {}

        if model_name not in tasks_inputs[task_name]:
            tasks_inputs[task_name][model_name] = df
        else:
            prev_task_length = len(tasks_inputs[task_name][model_name])
            curr_task_length = len(df)
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
                tasks_inputs[task_name][model_name] = df

    # Remove tasks where we do not have at least two players to rate
    for task_name in [task_name for task_name in tasks_inputs if len(tasks_inputs[task_name]) < 2]:
        log.warning("Removing task %s as there are not enough players.", task_name)
        del tasks_inputs[task_name]

    for task_name in tasks_inputs:
        online_ratings = {}
        final_ratings = {}

        # Set initial ratings
        for model_name in tasks_inputs[task_name]:
            online_ratings[model_name] = args.initial_rating

        # Get game matches
        task_matches = _sample_games(tasks_inputs[task_name], n=args.num_samples)

        # Show players coverage
        coverage = {model_name: 0 for model_name in tasks_inputs[task_name]}
        for task_match in task_matches:
            coverage[task_match["player_a_name"]] += 1
            coverage[task_match["player_b_name"]] += 1
        log.info("Player coverage: %s", coverage)

        # Evaluate win/draw/loss
        data = datasets.Dataset.from_list(task_matches)
        if args.criterion == "llama_score":
            from src.data.pipelines.text import elo_score_llama32

            data = data.map(
                elo_score_llama32,
                batched=True,
                batch_size=512,
                with_rank=True,
                fn_kwargs={
                    "prediction_a_column": "player_a_response",
                    "prediction_b_column": "player_b_response",
                },
            )

            scores = data["elo_score"]
            scores = [int(score) if score in ["0", "1"] else 0.5 for score in scores]
            data = data.remove_columns(["elo_score"])
            data = data.add_column("score", scores)

        elif args.criterion == "semantic_similarity":
            from src.data.pipelines.text import encode_sentence_bert

            data.set_format("torch")
            data = data.map(
                encode_sentence_bert,
                batched=True,
                batch_size=1024,
                fn_kwargs={"input_column": "player_a_response"},
            )
            data = data.map(
                encode_sentence_bert,
                batched=True,
                batch_size=1024,
                fn_kwargs={"input_column": "player_b_response"},
            )
            data = data.map(
                encode_sentence_bert,
                batched=True,
                batch_size=1024,
                fn_kwargs={"input_column": "reference"},
            )

            refs_z = data["reference_sentence_bert_embeds"].unsqueeze(1)
            player_a_preds_z = data["player_a_response_sentence_bert_embeds"].unsqueeze(2)
            player_b_preds_z = data["player_b_response_sentence_bert_embeds"].unsqueeze(2)

            player_a_scores = torch.bmm(refs_z, player_a_preds_z).squeeze()
            player_b_scores = torch.bmm(refs_z, player_b_preds_z).squeeze()
            scores_difference = player_a_scores - player_b_scores

            # Get scores (1: player a wins, 0.5: draw, 0: player a loses)
            threshold = 0.05
            scores = torch.zeros_like(scores_difference) + 0.5  # Default to draw
            scores[scores_difference > threshold] = 1
            scores[scores_difference < -threshold] = 0
            scores = scores.tolist()

            columns_to_remove = [
                "reference_sentence_bert_embeds",
                "player_a_response_sentence_bert_embeds",
                "player_b_response_sentence_bert_embeds",
            ]
            data = data.remove_columns(columns_to_remove)
            data = data.add_column("score", scores)
            data.set_format(None)

        else:
            raise ValueError("Unknown winning criterion %s", args.criterion)

        log.info("Scores counter: %s", Counter(scores))

        # Evaluate the online rating
        for row in data:
            player_a_new_rating, player_b_new_rating = _elo_rating(
                online_ratings[row["player_a_name"]],
                online_ratings[row["player_b_name"]],
                row["score"],
                k_factor=args.k_factor,
                zero_sum=not args.disable_zero_sum,
            )

            online_ratings[row["player_a_name"]] = player_a_new_rating
            online_ratings[row["player_b_name"]] = player_b_new_rating

        # Evaluate the final rating as the median of multiple rounds
        shuffled_data = data.shuffle()
        bootstrap_ratings = []
        for i in range(args.num_rounds):
            round_ratings = {}

            # Set initial ratings
            for model_name in tasks_inputs[task_name]:
                round_ratings[model_name] = args.initial_rating

            for row in shuffled_data.shard(args.num_rounds, i):
                player_a_new_rating, player_b_new_rating = _elo_rating(
                    round_ratings[row["player_a_name"]],
                    round_ratings[row["player_b_name"]],
                    row["score"],
                    k_factor=args.k_factor,
                    zero_sum=not args.disable_zero_sum,
                )

                round_ratings[row["player_a_name"]] = player_a_new_rating
                round_ratings[row["player_b_name"]] = player_b_new_rating

            bootstrap_ratings.append(round_ratings)

            for player in online_ratings:
                player_ratings = [round_ratings[player] for round_ratings in bootstrap_ratings]
                final_ratings[player] = np.median(player_ratings).item()

        leaderboard_repr = ""
        leaderboard_repr += f"Online Elo ratings on {task_name}:\n"
        online_leaderboard = dict(sorted(online_ratings.items(), key=lambda x: x[1], reverse=True))
        for i, key in enumerate(online_leaderboard):
            leaderboard_repr += (
                f"{str(i + 1) + '.':<3} {key:<29}: {int(online_leaderboard[key])}\n"
            )
        print(leaderboard_repr)

        leaderboard_repr = ""
        leaderboard_repr += f"Final Elo ratings on {task_name}:\n"
        final_leaderboard = dict(sorted(final_ratings.items(), key=lambda x: x[1], reverse=True))
        for i, key in enumerate(final_leaderboard):
            leaderboard_repr += f"{str(i + 1) + '.':<3} {key:<29}: {int(final_leaderboard[key])}\n"
        print(leaderboard_repr)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=str,
        help="Path to the folder containing the samples to process",
    )
    parser.add_argument(
        "-c",
        "--criterion",
        required=True,
        choices=["llama_score", "semantic_similarity"],
        type=str,
        help="Criterion to evaluate win/draw/loss",
    )
    parser.add_argument(
        "-r",
        "--initial-rating",
        default=1000,
        type=int,
        help="Initial Elo rating for new players (default: 1000)",
    )
    parser.add_argument(
        "-k",
        "--k-factor",
        default=16,
        type=int,
        help="Determines how much a single game affects the rating (default: 16)",
    )
    parser.add_argument(
        "-b",
        "--num-rounds",
        default=100,
        type=int,
        help="Number of rounds to bootstrap the final Elo rating (default 100)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        default=10_000,
        type=int,
        help="Number of games to sample to evaluate the score (default 10'000)",
    )
    parser.add_argument(
        "--disable-zero-sum",
        action="store_true",
        help="Whether to disable the zero-sum adjustment",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed for reproducibility (default: 1234)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (default: INFO)"
    )
    args = parser.parse_args()

    main(args)
