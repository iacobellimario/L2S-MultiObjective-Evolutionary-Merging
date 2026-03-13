import os
import json
from transformers import AutoTokenizer

def compute_avg_token_length(jsonl_path, tokenizer):
    """
    Compute the average token length of all samples in a .jsonl file
    """
    total_length = 0
    n_samples = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            code = data.get("code", [])

            if code:
                text = code[0] if isinstance(code, list) else code
                tokenized = tokenizer(text, return_tensors="pt")
                total_length += tokenized.input_ids.shape[1]
                n_samples += 1

    return total_length / n_samples if n_samples > 0 else 0.0


def extract_metrics(json_path):
    """
    Extract accuracy and number of samples from evaluator JSON
    """
    if not os.path.isfile(json_path):
        return None, None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        acc = data.get("acc")
        num_samples = data.get("num_samples")

        return acc, num_samples

    except json.JSONDecodeError:
        return None, None


def analyze_all_tasks(root_folder, tokenizer):
    """
    Analyze all benchmark tasks in root_folder, using evaluator sample counts
    """

    results = []

    sum_acc = 0
    sum_len = 0
    n_tasks = 0

    sum_acc_weighted = 0
    sum_len_weighted = 0
    total_samples = 0

    for task_name in sorted(os.listdir(root_folder)):

        task_path = os.path.join(root_folder, task_name)
        if not os.path.isdir(task_path):
            continue

        jsonl_files = [f for f in os.listdir(task_path) if f.endswith(".jsonl")]
        json_files = [f for f in os.listdir(task_path) if f.endswith(".json")]

        if not jsonl_files or not json_files:
            continue

        jsonl_path = os.path.join(task_path, jsonl_files[0])
        json_path = os.path.join(task_path, json_files[0])

        acc, num_samples = extract_metrics(json_path)

        if acc is None or num_samples is None:
            continue

        avg_len = compute_avg_token_length(jsonl_path, tokenizer)

        results.append((task_name, num_samples, acc, avg_len))

        # simple averages (unweighted)
        sum_acc += acc
        sum_len += avg_len
        n_tasks += 1

        # weighted averages (weighted by num_samples)
        sum_acc_weighted += acc * num_samples
        sum_len_weighted += avg_len * num_samples
        total_samples += num_samples

    # Print per-task results
    print(f"{'Benchmark':<20} {'Samples':<10} {'Accuracy':<10} {'Length':<10}")
    print("-" * 60)

    for task, n, acc, length in results:
        print(f"{task:<20} {n:<10} {acc:<10.1f} {length:<10.1f}")

    # Compute averages
    avg_acc = sum_acc / n_tasks if n_tasks else 0
    avg_len = sum_len / n_tasks if n_tasks else 0

    avg_acc_weighted = sum_acc_weighted / total_samples if total_samples else 0
    avg_len_weighted = sum_len_weighted / total_samples if total_samples else 0

    print("-" * 60)
    print(f"{'Average':<20} {'-':<10} {avg_acc:<10.1f} {avg_len:<10.1f}")
    print(f"{'Weighted Average':<20} {'-':<10} {avg_acc_weighted:<10.1f} {avg_len_weighted:<10.1f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder",
        type=str,
        required=True,
        help="Root folder containing benchmark subdirectories"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path or name of the tokenizer"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    analyze_all_tasks(args.root_folder, tokenizer)