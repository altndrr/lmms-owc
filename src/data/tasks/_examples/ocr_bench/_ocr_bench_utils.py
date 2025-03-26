__all__ = ["aggregate_accuracy", "doc_to_visual", "doc_to_text", "process_results"]


OCR_BENCH_SCORES = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}


def aggregate_accuracy(results: list, args: object) -> float:
    """Aggregate accuracy scores from OCR benchmark results and save to file.

    Args:
    ----
        results (list): List of dictionaries containing OCR benchmark results, each with
            'question_type' and 'score' keys.
        args (object): Arguments object containing output path and other configuration parameters.

    """
    for result in results:
        OCR_BENCH_SCORES[result["question_type"]] += result["score"]

    recognition_score = (
        OCR_BENCH_SCORES["Regular Text Recognition"]
        + OCR_BENCH_SCORES["Irregular Text Recognition"]
        + OCR_BENCH_SCORES["Artistic Text Recognition"]
        + OCR_BENCH_SCORES["Handwriting Recognition"]
        + OCR_BENCH_SCORES["Digit String Recognition"]
        + OCR_BENCH_SCORES["Non-Semantic Text Recognition"]
    )
    final_score = (
        recognition_score
        + OCR_BENCH_SCORES["Scene Text-centric VQA"]
        + OCR_BENCH_SCORES["Doc-oriented VQA"]
        + OCR_BENCH_SCORES["Key Information Extraction"]
        + OCR_BENCH_SCORES["Handwritten Mathematical Expression Recognition"]
    )

    return final_score / 1000


def doc_to_visual(doc: dict) -> list:
    """Process OCRBench document to extract visual data.

    Args:
    ----
        doc (dict): Dictionary containing OCRBench document data, expected to have an 'image' key
            with image data that can be converted to RGB format.

    """
    return [doc["image"].convert("RGB")]


def doc_to_text(doc: dict) -> str:
    """Process OCRBench document to extract question text.

    Args:
    ----
        doc (dict): Dictionary containing OCRBench document data with a 'question' key
            containing the question text to be processed.

    """
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}"


def process_results(doc: dict, results: list) -> dict:
    """Process OCRBench results by comparing predictions with ground truth answers.

    Args:
    ----
        doc (dict): Dictionary containing document information including:
            - answer: Ground truth answer (str or list)
            - dataset: Name of the dataset (str)
            - question_type: Type of OCR question (str)
        results (list): List containing model predictions as strings

    """
    pred = results[0].lower().strip()
    gt_ans = doc["answer"]
    dataset_name = doc["dataset"]

    score = 0
    if dataset_name == "HME100k":
        if isinstance(gt_ans, list):
            for j in range(len(gt_ans)):
                answer = gt_ans[j].strip().replace("\n", " ").replace(" ", "")
                predict = pred.strip().replace("\n", " ").replace(" ", "")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.strip().replace("\n", " ").replace(" ", "")
            predict = pred.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        if isinstance(gt_ans, list):
            for j in range(len(gt_ans)):
                answer = gt_ans[j].lower().strip().replace("\n", " ")
                predict = pred.lower().strip().replace("\n", " ")
                if answer in predict:
                    score = 1
        else:
            answer = gt_ans.lower().strip().replace("\n", " ")
            predict = pred.lower().strip().replace("\n", " ")
            if answer in predict:
                score = 1

    return {
        "ocr_bench_accuracy": {
            "question_type": doc["question_type"],
            "score": score,
            "prediction": pred,
            "ground_truth": gt_ans,
        },
    }
