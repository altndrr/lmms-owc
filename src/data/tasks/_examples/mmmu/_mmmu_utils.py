import ast
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests  # pytype: disable=pyi-error
import yaml
from openai import AzureOpenAI, OpenAI

from src import utils

__all__ = [
    "aggregate_judge_results",
    "aggregate_results",
    "doc_to_visual",
    "doc_to_text",
    "process_reasoning_results",
    "process_results",
]

log = utils.get_logger(__name__, rank_zero_only=True)

DOMAIN_CAT2SUB_CAT = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": [
        "Biology",
        "Chemistry",
        "Geography",
        "Math",
        "Physics",
    ],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": [
        "History",
        "Literature",
        "Sociology",
        "Psychology",
    ],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}


with open(Path(__file__).parent / "assets" / "_default_template_yaml") as f:
    raw_data = f.readlines()
    safe_data = []
    for _, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))

with open(Path(__file__).parent / "mmmu_val_reasoning.yaml") as f:
    raw_data = f.readlines()
    safe_data = []
    for _, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    reasoning_config = yaml.safe_load("".join(safe_data))
    MC_PROMPT = reasoning_config["model_specific_kwargs"]["default"]["multiple_choice_prompt"]
    OPEN_ENDED_PROMPT = reasoning_config["model_specific_kwargs"]["default"]["open_ended_prompt"]


NUM_SECONDS_TO_SLEEP = 5
API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4o-2024-08-06")

JUDGE_RULES = """You are a strict evaluator assessing answer correctness. You must output 1 for fully correct answers and 0 for any other case.
# Input
Question:
```
{question}
```
Ground Truth Answer:
```
{answer}
```
Model Prediction:
```
{pred}
```

# Evaluation Rules
- The model prediction contains the reasoning process, you should spot the final answer from the it.
- For multiple-choice questions: Score 1 if the predicted answer matches the correct answer.
- For open-ended questions:
  * Score 1 if the prediction matches the answer semantically and contains all key elements
  * Score 0 for partially correct answers or answers with extra incorrect information, even if the reasoning process is correct.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Strict Output format
[0/1]"""  # noqa: E501

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    client = OpenAI(api_key=API_KEY)
elif API_TYPE == "azure":
    API_URL = os.getenv(
        "AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken"
    )
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    client = AzureOpenAI(azure_endpoint=API_URL, api_version="2023-07-01-preview", api_key=API_KEY)


def aggregate_judge_results(results: list) -> float:
    """Aggregate the results of the judge.

    Args:
    ----
        results (list[str]): List of string predictions from the model, typically containing a
            single prediction

    """
    total_score = 0
    for result in results:
        try:
            total_score += int(result["score"])
        except:  # noqa: E722
            log.warning("Failed to convert score to int for %d: %f", result["id"], result["score"])
            total_score += 0
    return total_score / len(results)


def _get_chat_response(content: str, max_tokens: int, retries: int = 5) -> str:
    """Make an API call to chat model and get response with retry logic.

    Args:
    ----
        content (str): The prompt/content to send to the chat model
        max_tokens (int): Maximum number of tokens allowed in the response
        retries (int, optional): Number of retry attempts if the API call fails. Defaults to 5.

    """
    global MODEL_VERSION
    global client

    messages = [
        {
            "role": "system",
            "content": "You are a helpful and precise assistant for checking the correctness of the answer.",  # noqa: E501
        },
        {"role": "user", "content": content},
    ]

    payload = {
        "model": MODEL_VERSION,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content.strip()
            return content
        except requests.exceptions.RequestException as e:
            log.warning("Request failed on attempt %d: %s", attempt + 1, e)
            time.sleep(NUM_SECONDS_TO_SLEEP)
            if attempt == retries - 1:
                log.error("Failed to get response after %d attempts", retries)
                return ""
        except Exception as e:
            log.error("Error on attempt %d: %s", attempt + 1, e)
            return ""

    return ""


def _calculate_ins_level_acc(results: dict) -> float:
    """Calculate the instruction level accuracy for given Subject results.

    Args:
    ----
        results (dict): Dictionary containing results for each subject, where each entry has
            'acc' (accuracy) and 'num_example' (number of examples) fields.

    """
    acc = 0
    ins_num = 0
    for cat_results in results.values():
        acc += cat_results["acc"] * cat_results["num_example"]
        ins_num += cat_results["num_example"]
    if ins_num == 0:
        return 0
    return acc / ins_num


def _eval_multi_choice(gold_i: str | list[str], pred_i: str) -> bool:
    """Evaluate a multiple choice instance.

    Args:
    ----
        gold_i (str | list[str]): The correct answer(s). Can be either a single string or a list
            of strings.
        pred_i (str): The predicted answer as a string.

    """
    correct = False
    # Only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def _eval_open(gold_i: str | list[str], pred_i: list[str | float]) -> bool:
    """Evaluate an open question instance.

    Args:
    ----
        gold_i (str | list[str]): The correct answer(s). Can be either a single string or a list
            of strings.
        pred_i (list[str | float]): A list of predicted answers, where each answer can be either a
            string or float.

    """
    correct = False
    if isinstance(gold_i, list):
        # Use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(_normalize_str(answer))
    else:
        norm_answers = _normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # If it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # Only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # It's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def _evaluate_mmmu(samples: list[dict]) -> tuple[dict, dict]:
    """Batch evaluation for multiple choice and open questions.

    Args:
    ----
        samples (list[dict]): A list of dictionaries where each dictionary represents a sample
            containing keys for 'id', 'answer', 'parsed_pred', and 'question_type'

    """
    pred_correct = 0
    judge_dict = dict()
    for sample in samples:
        gold_i = sample["answer"]
        pred_list = sample["parsed_pred"]
        correct = False
        for pred_i in pred_list:
            if sample["question_type"] == "multiple-choice":
                correct = _eval_multi_choice(gold_i, pred_i)
            else:  # open question
                correct = _eval_open(gold_i, pred_i)

            if correct:
                judge_dict[sample["id"]] = "Correct"
                pred_correct += 1
                break

        if not correct:
            judge_dict[sample["id"]] = "Wrong"

    if len(samples) == 0:
        return judge_dict, {"acc": 0}

    return judge_dict, {"acc": pred_correct / len(samples)}


def aggregate_results(results: list[dict]) -> float:
    """Aggregate evaluation results across multiple domains and calculate overall accuracy.

    Args:
    ----
        results (list[dict]): A list of dictionaries containing evaluation results. Each dictionary
            should contain keys for subdomain information and evaluation metrics.

    """
    evaluation_result = {}
    subset_to_eval_samples = defaultdict(list)
    for result in results:
        subset_to_eval_samples[result["subdomain"]].append(result)
    for subset, sub_eval_samples in subset_to_eval_samples.items():
        _, metric_dict = _evaluate_mmmu(sub_eval_samples)
        metric_dict.update({"num_example": len(sub_eval_samples)})
        evaluation_result[subset] = metric_dict
    printable_results = {}
    for domain, in_domain_cats in DOMAIN_CAT2SUB_CAT.items():
        in_domain_cat_results = {}
        for cat_name in in_domain_cats:
            if cat_name in evaluation_result:
                in_domain_cat_results[cat_name] = evaluation_result[cat_name]
            else:
                pass
        in_domain_ins_acc = _calculate_ins_level_acc(in_domain_cat_results)
        in_domain_data_num = sum(
            [cat_results["num_example"] for cat_results in in_domain_cat_results.values()]
        )
        printable_results["Overall-" + domain] = {
            "num": int(in_domain_data_num),
            "acc": round(in_domain_ins_acc, 5),
        }

        # Add sub category
        for cat_name, cat_results in in_domain_cat_results.items():
            printable_results[cat_name] = {
                "num": int(cat_results["num_example"]),
                "acc": round(cat_results["acc"], 5),
            }
    all_ins_acc = _calculate_ins_level_acc(evaluation_result)
    printable_results["Overall"] = {
        "num": sum([cat_results["num_example"] for cat_results in evaluation_result.values()]),
        "acc": round(all_ins_acc, 5),
    }

    return printable_results["Overall"]["acc"]


def _parse_options(options: list[str]) -> str:
    """Format a list of multiple choice options into a string with letter prefixes.

    Args:
    ----
        options (list[str]): List of option strings to be formatted with letter prefixes,
                starting from 'A'

    """
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options, strict=True)
        ]
    )
    return choices_str


def _construct_prompt(
    doc: dict, multi_choice_prompt: str = "", open_ended_prompt: str = ""
) -> str:
    """Construct a formatted prompt from the document data.

    Args:
    ----
        doc (dict): A dictionary containing question data with required keys:
            - 'question': The question text
            - 'question_type': Type of question ('multiple-choice' or 'open-ended')
            - 'options': List of options (required for multiple-choice questions)
        multi_choice_prompt (str, optional): Template prompt for multiple choice questions.
            Defaults to "".
        open_ended_prompt (str, optional): Template prompt for open ended questions.
            Defaults to "".

    """
    question = doc["question"]
    if doc["question_type"] == "multiple-choice":
        # Weirdly, data["options"] is a string in MMMU Huggingface dataset
        parsed_options = _parse_options(ast.literal_eval(doc["options"]))
        # parsed_options already prepends a newline so no need to add space here
        question = f"{question}\n{parsed_options}\n\n{multi_choice_prompt}"
    else:
        question = f"{question}\n\n{open_ended_prompt}"
    return question


def _replace_images_tokens(input_string: str) -> str:
    """Replace the numbered image tokens with a generic image token.

    Args:
    ----
        input_string (str): A string that may contain numbered image tokens in the format
            '<image n>' where n is a number from 1 to 7

    """
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def doc_to_text(doc: dict, model_specific_kwargs: dict) -> str:
    """Convert a document dictionary into a formatted text prompt.

    Args:
    ----
        doc (dict): A dictionary containing question data and metadata, with keys such as
             'question', 'question_type', and 'options' for multiple choice questions.
        model_specific_kwargs (dict): Extra kwargs about the model.

    """
    question = _construct_prompt(
        doc,
        model_specific_kwargs.get("multi_choice_prompt"),
        model_specific_kwargs.get("open_ended_prompt"),
    )
    if config["metadata"]["interleaved_format"]:
        question = _replace_images_tokens(question)
    return question


def doc_to_visual(doc: dict) -> list:
    """Extract and process visual elements from a document.

    Args:
    ----
        doc (dict): A dictionary containing question data and image information.
            Expected to contain image tokens as keys and corresponding PIL images as values.

    """
    prompt = _construct_prompt(doc)
    image_tokens = re.findall(r"<image \d+>", prompt)
    # Remove <> and  swap space as _
    image_tokens = sorted(
        list(set([image_token.strip("<>").replace(" ", "_") for image_token in image_tokens]))
    )
    visual = [doc[image_token].convert("RGB") for image_token in image_tokens]
    return visual


def _extract_subset_name(input_string: str) -> str:
    """Extract the subset name from a given validation string.

    Args:
    ----
        input_string (str): A string following the pattern 'split_subsetname_number',
            where split is typically 'validation' or similar prefix

    """
    split = input_string.split("_")[0]
    pattern = re.compile(rf"^{split}_(.+?)_\d+$")
    match = pattern.search(input_string)
    if match:
        return match.group(1)
    else:
        raise ValueError(f'No match found in "{input_string}"')


def _extract_numbers(string: str) -> list[str]:
    """Extract all forms of numbers from a string with regex.

    Based on implementation from MMMU repository:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L100

    Args:
    ----
        string (str): Input string to extract numbers from. Can contain numbers in various formats:
            - Numbers with commas (e.g., "1,000")
            - Scientific notation (e.g., "1.23e-4")
            - Simple numbers (e.g., "123", "0.123")

    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def _get_multi_choice_info(options: list[str]) -> tuple[dict[str, str], list[str]]:
    """Given the list of options for multiple choice question.

    Based on MMMU repository implementation:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54

    Args:
    ----
        options (list[str]): A list of strings representing multiple choice options to be mapped
            to letters starting from 'A'

    """
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


def _check_is_number(string: str) -> bool:
    """Check if the given string is a number.

    Based on implementation from MMMU repository:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L65

    Args:
    ----
        string (str): Input string to be checked for numeric value. Commas are allowed
            within the number (e.g., "1,000").

    """
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        return False


def _normalize_str(string: str) -> list[str | float]:
    """Normalize a string by converting numbers to floats and standardizing text.

    Based on implementation from MMMU repository:
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L76

    Args:
    ----
        string (str): Input string to be normalized. Can contain either text or numeric values,
            including numbers with commas (e.g., "1,000").

    """
    string = string.strip()

    is_number = _check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        string = round(string, 2)  # Leave 2 decimal
        return [string]
    else:
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # Avoid trivial matches
        return [string]


def _parse_open_response(response: str) -> list[str | float]:
    """Parse the prediction from the generated response for open-ended questions.

    Args:
    ----
        response (str): The model's generated response text to be parsed. The response may contain
            multiple sentences, numbers, or equations that need to be extracted and normalized.

    """

    # Rest of the implementation remains unchanged
    def get_key_sub_responses(response: str) -> list[str]:
        key_responses = []
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # If last one, accept it's an equation (the entire response can be just one sentence
            # with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            # The shortest response that may contain the answer (tail part of the response)
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    else:
                        if len(resp.split(indicator)[-1].strip()) < len(shortest_key_response):
                            shortest_key_response = resp.split(indicator)[-1].strip()

            symbols = [":", ",", ".", "!", "?", ";", ":", "'"]
            if shortest_key_response and shortest_key_response.strip() not in symbols:
                key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # Did not found any
            return [response]
        return key_responses

    key_responses = get_key_sub_responses(response)

    pred_list = key_responses.copy()  # Keep the original string response
    for resp in key_responses:
        pred_list.extend(_extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(_normalize_str(pred_list[i]))
    pred_list = tmp_pred_list

    # Remove duplicates
    pred_list = list(set(pred_list))

    return pred_list


def _parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str:
    """Parse the prediction from the generated response for multiple choice questions.

    Args:
    ----
        response (str): The model's generated response text to be parsed
        all_choices (list[str]): List of available choice letters (e.g., ['A', 'B', 'C', 'D'])
        index2ans (dict[str, str]): Dictionary mapping choice letters to their corresponding answer
            text.

    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # Add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # If all above doesn't get candidates, check if the content is larger than 5 tokens and try to
    # parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # Still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)  # noqa: S311 # nosec: B311
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # Get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # If only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def process_reasoning_results(doc: dict, results: list[str]) -> dict:
    """Process evaluation results for a given document and model predictions.

    Args:
    ----
        doc (dict): Dictionary containing the question document with fields like 'question_type',
             'options', 'id', and 'answer'
        results (list[str]): List of string predictions from the model, typically containing a
            single prediction

    """
    pred = results[0]
    formatted_question = _construct_prompt(doc, MC_PROMPT, OPEN_ENDED_PROMPT)
    llm_judge_prompt = JUDGE_RULES.format(
        question=formatted_question, answer=doc["answer"], pred=pred
    )
    llm_judge_score = _get_chat_response(llm_judge_prompt, max_tokens=20, retries=3)
    mmmu_judge_acc = {
        "id": doc["id"],
        "subdomain": _extract_subset_name(doc["id"]),
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "pred": pred,
        "score": llm_judge_score,
    }
    return {"mmmu_judge_acc": mmmu_judge_acc}


def process_results(doc: dict, results: list[str]) -> dict:
    """Process evaluation results for a given document and model predictions.

    Args:
    ----
        doc (dict): Dictionary containing the question document with fields like 'question_type',
             'options', 'id', and 'answer'
        results (list[str]): List of string predictions from the model, typically containing a
            single prediction

    """
    parsed_preds = []
    for pred in results:
        if doc["question_type"] == "multiple-choice":
            index2ans, all_choices = _get_multi_choice_info(ast.literal_eval(doc["options"]))
            parsed_pred = _parse_multi_choice_response(pred, all_choices, index2ans)
        else:
            parsed_pred = _parse_open_response(pred)

        parsed_preds.append(parsed_pred)

    mmmu_exact_acc = {
        "id": doc["id"],
        "subdomain": _extract_subset_name(doc["id"]),
        "question_type": doc["question_type"],
        "answer": doc["answer"],
        "parsed_pred": parsed_preds,
    }
    return {"mmmu_acc": mmmu_exact_acc, "mmmu_acc_pass_at_k": mmmu_exact_acc}
