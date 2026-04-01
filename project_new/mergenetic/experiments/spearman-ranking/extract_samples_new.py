# ==== Imports ====
import os
import json
import torch
import random
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import spearmanr

# pymoo components (kept as-is)
from mergenetic.merging.taskarithmetic_merger import TaskArithmeticMerger

# Mergekit and Mergenetic
from mergenetic.utils import ConfigLmEval
from mergenetic import PROJECT_ROOT

# Hugging Face
from huggingface_hub import snapshot_download

method_to_label = {
    "random": "Random",
    "disagreement": "Disagreement",
    "entropy": "Entropy",
}

# ==== Plot palette ====
METHOD_TO_COLOR = {
    "random": "#0F60BD",
    "entropy": "#C62828",
    "disagreement": "#05564A",
}

# ==== Set the seed and max number of evaluated samples ====
SEED = 42
NUM_SAMPLES = 1000

PLOT_ALPHA = 0.9


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute ranking of merged models with different genotypes and extract samples that maximize entropy."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/extracted_samples.csv",
        help="Output file to store extracted sample ids.",
    )
    parser.add_argument(
        "--evaluation_dir",
        type=str,
        default="outputs",
        help="Directory to store evaluation results.",
    )
    parser.add_argument("--task", type=str, default="gsm8k", help="Task name for evaluation.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="Base model name.",
    )
    parser.add_argument(
        "--reasoning_model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Reasoning model name.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation.")
    parser.add_argument("--num_genotypes", type=int, default=2, help="Number of genotypes to evaluate.")
    parser.add_argument("--run_id", type=str, default="gsm8k_1.5B")
    return parser.parse_args()


# =========================================================
# Exporters
# =========================================================
def export_spearman_table(
    plot_data_df: pd.DataFrame,
    run_id: str,
    plot_type: str,
    output_dir: str = "tables/spearman",
    filename_suffix: str = "",
):
    """
    Export LaTeX table with Spearman correlation (mean ± CI)
    for a given plot_type.

    plot_type examples:
      - "sample_for_accuracy"
      - "sample_for_accuracy_test_on_length"
    """
    os.makedirs(output_dir, exist_ok=True)

    df = plot_data_df[plot_data_df["plot_type"] == plot_type].copy()
    if df.empty:
        print(f"⚠️ No rows found for plot_type='{plot_type}'. Skipping export.")
        return

    # convert sampling percentage to integer %
    df["sample_percent"] = (df["sample_percent"] * 100).round().astype(int)

    # format mean ± CI (LaTeX-friendly)
    df["value"] = df.apply(
        lambda r: f"{r['mean_correlation']:.3f}\\pm{r['ci_correlation']:.3f}",
        axis=1,
    )

    # robust: supports single/multiple benchmarks
    table = (
        df.pivot_table(
            index=["benchmark", "method"],
            columns="sample_percent",
            values="value",
            aggfunc="first",
        )
        .sort_index()
    )

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    tex_path = os.path.join(output_dir, f"spearman_{run_id}{suffix}.tex")

    with open(tex_path, "w") as f:
        f.write(
            table.to_latex(
                escape=False,
                multirow=True,
                column_format="ll" + "c" * len(table.columns),
            )
        )

    print(f"✅ Exported LaTeX table: {tex_path}")


# =========================================================
# IO / data helpers
# =========================================================
def extract_index_and_score(file_path, bench_name, genotype):
    """
    Reads a JSONL file and extracts 'idx' and 'score' from each line.
    Also stores 'answer_length' from len(data['code'][0]) when present.
    """
    results = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "idx" in data and "score" in data:
                        results.append(
                            {
                                "index": data["idx"],
                                "score": data["score"][0],
                                "benchmark": bench_name,
                                "genotype": genotype,
                                "answer_length": len(data["code"][0]) if "code" in data else None,
                            }
                        )
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:120]}...")
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")

    return results


def update_accuracies(performance_df, correctness_df, benchmarks, genotypes):
    """
    Update the 'accuracy' column in performance_df with recalculated values from correctness_df.
    Also adds 'mean_answer_length' and min-max normalizes it per benchmark over genotypes.
    """
    for benchmark in benchmarks:
        n_items = len(correctness_df[correctness_df["benchmark"] == benchmark]["index"].unique())
        if n_items == 0:
            continue

        for genotype in genotypes:
            acc = (
                correctness_df[
                    (correctness_df["benchmark"] == benchmark) & (correctness_df["genotype"] == genotype)
                ]["score"].sum()
                / n_items
            )
            performance_df.loc[
                (performance_df["benchmark"] == benchmark) & (performance_df["genotype"] == genotype),
                "accuracy",
            ] = acc

        # mean lengths
        for genotype in genotypes:
            mean_len = correctness_df[
                (correctness_df["benchmark"] == benchmark) & (correctness_df["genotype"] == genotype)
            ]["answer_length"].mean()
            performance_df.loc[
                (performance_df["benchmark"] == benchmark) & (performance_df["genotype"] == genotype),
                "mean_answer_length",
            ] = mean_len

        # normalize mean_answer_length per benchmark (min-max across genotypes)
        max_length = performance_df[performance_df["benchmark"] == benchmark]["mean_answer_length"].max()
        min_length = performance_df[performance_df["benchmark"] == benchmark]["mean_answer_length"].min()

        if pd.notna(max_length) and pd.notna(min_length) and max_length != min_length:
            performance_df.loc[performance_df["benchmark"] == benchmark, "mean_answer_length"] = (
                performance_df.loc[performance_df["benchmark"] == benchmark, "mean_answer_length"] - min_length
            ) / (max_length - min_length)
        else:
            performance_df.loc[performance_df["benchmark"] == benchmark, "mean_answer_length"] = np.nan

    return performance_df


# =========================================================
# Sampling strategies
# =========================================================
def get_random_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score="score"):
    random.seed(seed)
    idxes_per_benchmark = {
        benchmark: correctness_df[correctness_df["benchmark"] == benchmark]["index"].unique().tolist()
        for benchmark in benchmarks
    }
    all_sampled_idx = {}
    for benchmark in benchmarks:
        sample_size = max(1, int(sample_percent * len(idxes_per_benchmark[benchmark])))
        sampled_idxes = random.sample(idxes_per_benchmark[benchmark], sample_size)
        all_sampled_idx[benchmark] = sampled_idxes
    return all_sampled_idx


def get_disagreement_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score="score"):
    random.seed(seed)
    all_sampled_idx = {}

    if not all(g in correctness_df["genotype"].unique() for g in [0.0, 1.0]):
        raise ValueError("Both genotypes 0.0 and 1.0 must be present in correctness_df for disagreement sampling.")

    for benchmark in benchmarks:
        idxes = correctness_df[correctness_df["benchmark"] == benchmark]["index"].unique().tolist()

        scores_0 = correctness_df[
            (correctness_df["benchmark"] == benchmark) & (correctness_df["genotype"] == 0.0)
        ].set_index("index")[score]

        scores_1 = correctness_df[
            (correctness_df["benchmark"] == benchmark) & (correctness_df["genotype"] == 1.0)
        ].set_index("index")[score]

        disagreement_indices = []
        for idx in idxes:
            if idx in scores_0.index and idx in scores_1.index:
                if score == "answer_length":
                    # heuristic threshold for length disagreement
                    denom = scores_0.loc[idx] if scores_0.loc[idx] != 0 else 1e-9
                    if (scores_0.loc[idx] - scores_1.loc[idx]) / denom > 0.5:
                        disagreement_indices.append(idx)
                else:
                    if scores_0.loc[idx] != scores_1.loc[idx]:
                        disagreement_indices.append(idx)

        sample_size = max(1, int(sample_percent * len(idxes)))
        if len(disagreement_indices) >= sample_size:
            sampled_idxes = random.sample(disagreement_indices, sample_size)
        else:
            sampled_idxes = disagreement_indices.copy()
            remaining = [idx for idx in idxes if idx not in disagreement_indices]
            additional_needed = sample_size - len(disagreement_indices)
            additional = random.sample(remaining, min(additional_needed, len(remaining)))
            sampled_idxes.extend(additional)
            if len(sampled_idxes) < sample_size:
                print(
                    f"Warning: Not enough indices for benchmark {benchmark}, "
                    f"sampled {len(sampled_idxes)} / {sample_size}"
                )

        all_sampled_idx[benchmark] = sampled_idxes

    return all_sampled_idx


def get_entropy_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score="score"):
    """
    Entropy over Bernoulli correctness across genotypes.
    Excludes 1 genotype randomly each run to add variance across folds.
    """
    random.seed(seed)
    all_sampled_idx = {}

    genotypes = correctness_df["genotype"].unique().tolist()
    if len(genotypes) <= 1:
        raise ValueError("Need at least 2 genotypes to compute entropy sampling.")

    # exclude one random genotype
    sampled_genotypes = random.sample(genotypes, k=len(genotypes) - 1)

    for benchmark in benchmarks:
        df_b = correctness_df[correctness_df["benchmark"] == benchmark]
        df_b = df_b[df_b["genotype"].isin(sampled_genotypes)]

        grouped = df_b.groupby("index")[score].agg(["mean", "count"])
        p1 = grouped["mean"]
        p0 = 1 - p1

        # avoid log2(0)
        term1 = np.zeros_like(p1, dtype=float)
        term2 = np.zeros_like(p0, dtype=float)
        valid_p1 = p1 > 0
        valid_p0 = p0 > 0

        term1[valid_p1] = p1[valid_p1] * np.log2(p1[valid_p1])
        term2[valid_p0] = p0[valid_p0] * np.log2(p0[valid_p0])

        entropy = -(term1 + term2)
        entropy_dict = dict(zip(grouped.index, entropy))

        idxes = df_b["index"].unique()
        for idx in idxes:
            if idx not in entropy_dict:
                entropy_dict[idx] = 0.0

        high_entropy_indices = sorted(idxes, key=lambda x: entropy_dict[x], reverse=True)

        sample_size = max(1, int(sample_percent * len(idxes)))
        sampled_idxes = (
            high_entropy_indices[:sample_size]
            if len(high_entropy_indices) >= sample_size
            else high_entropy_indices
        )

        if len(sampled_idxes) < sample_size:
            remaining = [idx for idx in idxes if idx not in sampled_idxes]
            additional_needed = sample_size - len(sampled_idxes)
            sampled_idxes.extend(random.sample(remaining, min(additional_needed, len(remaining))))

        all_sampled_idx[benchmark] = sampled_idxes

    return all_sampled_idx


# =========================================================
# Spearman computation
# =========================================================
def compute_sample_accuracies(performance_df, correctness_df, benchmarks, genotypes, all_sampled_idx):
    perf_df = performance_df.copy()
    perf_df = perf_df[perf_df["benchmark"].isin(benchmarks) & perf_df["genotype"].isin(genotypes)]

    for benchmark in benchmarks:
        sampled_idxes = all_sampled_idx.get(benchmark, [])
        if len(sampled_idxes) == 0:
            continue

        for genotype in genotypes:
            sub = correctness_df[
                (correctness_df["benchmark"] == benchmark)
                & (correctness_df["genotype"] == genotype)
                & (correctness_df["index"].isin(sampled_idxes))
            ]

            if sub.empty:
                continue

            sample_score = sub["score"].sum() / len(sampled_idxes)
            sample_mean_length = sub["answer_length"].sum() / len(sampled_idxes)

            perf_df.loc[
                (perf_df["benchmark"] == benchmark) & (perf_df["genotype"] == genotype), "sample_acc"
            ] = sample_score
            perf_df.loc[
                (perf_df["benchmark"] == benchmark) & (perf_df["genotype"] == genotype), "sample_length"
            ] = sample_mean_length

    return perf_df


def compute_spearman_correlations(perf_df, benchmarks, col1="accuracy", col2="sample_acc"):
    spearman_correlations = {}
    for benchmark in benchmarks:
        col1_scores = perf_df[perf_df["benchmark"] == benchmark][col1]
        col2_scores = perf_df[perf_df["benchmark"] == benchmark][col2]

        if col1_scores.empty or col2_scores.empty:
            continue

        if col1_scores.nunique() <= 1 or col2_scores.nunique() <= 1:
            corr = np.nan
        else:
            corr, _ = spearmanr(col1_scores, col2_scores)

        spearman_correlations[benchmark] = corr

    return spearman_correlations


def plot_spearman_vs_sampling(
    performance_df,
    correctness_df,
    benchmarks,
    genotypes,
    val_genotypes,
    seed,
    k=5,
    sampling_methods=None,
    run_id=None,
    plot_dir="plots",
):
    """
    Produces a 2-column plot for each benchmark:
      (1) sample for accuracy -> test on accuracy
      (2) sample for accuracy -> test on length

    Also returns plot_data_df used for LaTeX table export.
    """
    from scipy.stats import t

    if sampling_methods is None:
        sampling_methods = {"random": get_random_sampled_indices}

    # fixed sampling grid — precompute integer tick values once
    sample_percents = np.linspace(0.05, 1.0, 10)
    tick_vals = (sample_percents * 100).round().astype(int)

    spearman_results_acc = {m: {b: [] for b in benchmarks} for m in sampling_methods}
    spearman_results_acc_for_len = {m: {b: [] for b in benchmarks} for m in sampling_methods}
    sampled_ids = []
    plot_data = []

    # ----------------------------
    # Spearman computation
    # ----------------------------
    for method_name, method_func in sampling_methods.items():
        for sp in sample_percents:
            fold_corrs_acc = {b: [] for b in benchmarks}
            fold_corrs_acc_len = {b: [] for b in benchmarks}

            for fold in range(k):
                fold_seed = seed + fold

                val_df = correctness_df[correctness_df["genotype"].isin(val_genotypes)]

                sampled_ids_acc = method_func(val_df, benchmarks, fold_seed, sp)
                sampled_ids_len = method_func(val_df, benchmarks, fold_seed, sp, score="answer_length")

                for b in benchmarks:
                    sampled_ids.append(
                        {
                            "method": method_name,
                            "sample_percent": sp,
                            "fold": fold,
                            "seed": fold_seed,
                            "benchmark": b,
                            "score": "accuracy",
                            "sampled_ids": sampled_ids_acc[b],
                        }
                    )
                    sampled_ids.append(
                        {
                            "method": method_name,
                            "sample_percent": sp,
                            "fold": fold,
                            "seed": fold_seed,
                            "benchmark": b,
                            "score": "answer_length",
                            "sampled_ids": sampled_ids_len[b],
                        }
                    )

                perf_acc = compute_sample_accuracies(
                    performance_df.copy(),
                    correctness_df,
                    benchmarks,
                    genotypes,
                    sampled_ids_acc,
                )

                corr_acc = compute_spearman_correlations(perf_acc, benchmarks, "accuracy", "sample_acc")
                corr_acc_len = compute_spearman_correlations(
                    perf_acc,
                    benchmarks,
                    "mean_answer_length",
                    "sample_length",
                )

                for b in benchmarks:
                    fold_corrs_acc[b].append(corr_acc.get(b, np.nan))
                    fold_corrs_acc_len[b].append(corr_acc_len.get(b, np.nan))

            def mean_ci(values):
                values = np.asarray(values, dtype=float)
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    return np.nan, np.nan
                if len(values) == 1:
                    return float(values[0]), 0.0
                mean = float(values.mean())
                std = float(values.std(ddof=1))
                ci = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
                return mean, ci

            for b in benchmarks:
                spearman_results_acc[method_name][b].append(mean_ci(fold_corrs_acc[b]))
                spearman_results_acc_for_len[method_name][b].append(mean_ci(fold_corrs_acc_len[b]))

    # ----------------------------
    # Plotting
    # ----------------------------
    fig, axes = plt.subplots(
        nrows=len(benchmarks),
        ncols=2,
        figsize=(8, 4.5 * len(benchmarks)),
        squeeze=False,
    )

    # keep stable legend order
    plot_order = [m for m in ["entropy", "disagreement", "random"] if m in sampling_methods]

    int_formatter = mticker.FuncFormatter(lambda x, _: f"{int(x)}")

    for i, b in enumerate(benchmarks):
        ax_acc = axes[i, 0]
        ax_len = axes[i, 1]

        for method_name in plot_order:
            color = METHOD_TO_COLOR.get(method_name, "black")

            means_acc = [spearman_results_acc[method_name][b][j][0] for j in range(len(sample_percents))]
            cis_acc   = [spearman_results_acc[method_name][b][j][1] for j in range(len(sample_percents))]
            means_len = [spearman_results_acc_for_len[method_name][b][j][0] for j in range(len(sample_percents))]
            cis_len   = [spearman_results_acc_for_len[method_name][b][j][1] for j in range(len(sample_percents))]

            # collect table rows
            for j, sp in enumerate(sample_percents):
                plot_data.append(
                    {
                        "benchmark": b,
                        "method": method_name,
                        "sample_percent": sp,
                        "plot_type": "sample_for_accuracy",
                        "mean_correlation": means_acc[j],
                        "ci_correlation": cis_acc[j],
                    }
                )
                plot_data.append(
                    {
                        "benchmark": b,
                        "method": method_name,
                        "sample_percent": sp,
                        "plot_type": "sample_for_accuracy_test_on_length",
                        "mean_correlation": means_len[j],
                        "ci_correlation": cis_len[j],
                    }
                )

            ax_acc.errorbar(
                tick_vals,
                means_acc,
                yerr=cis_acc,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                capsize=5,
                capthick=1.2,
                elinewidth=1.2,
                color=color,
                alpha=PLOT_ALPHA,
                label=method_to_label.get(method_name, method_name),
            )

            ax_len.errorbar(
                tick_vals,
                means_len,
                yerr=cis_len,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=5,
                capsize=5,
                capthick=1.2,
                elinewidth=1.2,
                color=color,
                alpha=PLOT_ALPHA,
                label=method_to_label.get(method_name, method_name),
            )

        for ax, title in (
            (ax_acc, "Ranking by Accuracy"),
            (ax_len, "Ranking by Length"),
        ):
            ax.set_title(title)
            ax.set_xlabel("Sample Percentage (%)")
            ax.set_xticks(tick_vals)
            ax.xaxis.set_major_formatter(int_formatter)
            ax.set_ylabel("Spearman Correlation")
            ax.set_ylim(0.0, 1.2)
            #ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
            ax.legend(loc="lower right")

    os.makedirs(plot_dir, exist_ok=True)
    plt.tight_layout()
    name = run_id if run_id is not None else "spearman"
    out_path = os.path.join(plot_dir, f"{name}_spearman_vs_sampling.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")

    plot_data_df = pd.DataFrame(plot_data)
    return spearman_results_acc, spearman_results_acc_for_len, sampled_ids, plot_data_df


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    if args.num_genotypes < 2:
        raise ValueError("num_genotypes must be at least 2 to perform evaluation.")

    # Reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    print(f"All seeds set to {SEED}")

    # directory where to store base and merged models
    model_dir = "/leonardo_scratch/fast/IscrC_LENS/miacobel/model"
    os.makedirs(model_dir, exist_ok=True)

    config = ConfigLmEval()
    config.additional_templates_folder = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", "lm_tasks")
    config.bench = args.task
    config.path_to_store_config = os.path.join(
        PROJECT_ROOT, "project_new", "mergenetic", "experiments", "spearman-ranking", "configs"
    )
    config.seed = SEED
    config.device = args.device
    config.run_id = args.run_id
    config.metric = "exact_match"
    config.task_type = "FG_MATH"
    config.path_to_store_merged_model = f"{model_dir}/merged"
    config.base_model = f"{model_dir}/{args.reasoning_model.split('/')[-1]}"
    config.models = {"en": f"{model_dir}/{args.base_model.split('/')[-1]}"}
    config.eval_batch_size = 8

    print(config)

    path_to_store_yaml = f"{config.path_to_store_config}/{config.run_id}"
    lang_id = "en"

    merger = TaskArithmeticMerger(
        run_id=config.run_id,
        path_to_base_model=config.base_model,
        model_paths=[config.models[lang_id]],
        path_to_store_yaml=path_to_store_yaml,
        path_to_store_merged_model=config.path_to_store_merged_model,
        dtype=config.dtype,
    )

    print("✅ Merger configured.")

    genotypes = list(np.linspace(0.0, 1.0, num=args.num_genotypes))

    # for each genotype, create a merged model and evaluate it
    for index, row in enumerate(genotypes):
        genotype = [row]
        print(f"Processing genotype {genotype} ({index+1}/{len(genotypes)})")

        metrics_path = os.path.join(
            PROJECT_ROOT,
            "project_new",
            "mergenetic",
            args.evaluation_dir,
            str(genotype[0]),
            f"{args.task}",
            "test_deepseek-math_1000_seed42_t0_s0_e-1_deepseek-math_metrics.json",
        )

        if os.path.exists(metrics_path):
            print(f"Results for genotype {genotype[0]} already exist. Skipping evaluation.")
            continue

        print(f"Evaluation not found at {metrics_path}. Proceeding with evaluation.")

        merged_model_conf_path = merger.create_individual_configuration(genotype)
        merged_model_path = merger.merge_model_from_configuration(merged_model_conf_path)
        print(f"✅ Merged model created at {merged_model_path}")

        subprocess.run(
            [
                "python3",
                "../Qwen2.5-Math/evaluation/math_eval.py",
                "--model_name_or_path", f"{merged_model_path}",
                "--data_names", f"{args.task}",
                "--output_dir", f"{args.evaluation_dir}/{genotype[0]}",
                "--prompt_type", "deepseek-math",
                "--n_sampling", "1",
                "--max_tokens_per_call", "10240",
                "--num_shots", "0",
                "--seed", f"{SEED}",
                "--split", "test",
                "--num_test_sample", f"{NUM_SAMPLES}",
                "--use_vllm",
                "--save_outputs",
                "--overwrite",
                "--data_dir", "../Qwen2.5-Math/evaluation/data",
            ],
            env={**os.environ},
            cwd=f"{PROJECT_ROOT}/project_new/mergenetic",
        )

    # After evaluating all genotypes, extract accuracy from metrics files
    results = []
    for genotype in genotypes:
        results_file = os.path.join(
            PROJECT_ROOT,
            "project_new",
            "mergenetic",
            args.evaluation_dir,
            str(genotype),
            f"{args.task}",
            "test_deepseek-math_1000_seed42_t0_s0_e-1_deepseek-math_metrics.json",
        )
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
            accuracy = data.get("acc", None)
            if accuracy is not None:
                results.append({"genotype": genotype, "benchmark": args.task, "accuracy": accuracy})
            else:
                print(f"No accuracy found in {results_file}")
        else:
            print(f"Results file {results_file} does not exist.")

    performance_df = pd.DataFrame(results)

    # Load correctness JSONL to compute per-item correctness + length
    results_correctness = []
    for genotype in genotypes:
        results_file = os.path.join(
            PROJECT_ROOT,
            "project_new",
            "mergenetic",
            args.evaluation_dir,
            str(genotype),
            f"{args.task}",
            f"test_deepseek-math_1000_seed{SEED}_t0_s0_e-1.jsonl",
        )
        correctness = extract_index_and_score(results_file, args.task, genotype)
        print(f"Extracted {len(correctness)} correctness entries from {results_file}")
        results_correctness.extend(correctness)

    correctness_df = pd.DataFrame(results_correctness)

    # update performance_df 'accuracy' and add normalized 'mean_answer_length'
    performance_df = update_accuracies(performance_df, correctness_df, [args.task], genotypes)

    # Compute Spearman + plots
    sp_results_acc, sp_results_len, sampled_ids, plot_data_df = plot_spearman_vs_sampling(
        performance_df,
        correctness_df,
        [args.task],
        genotypes,
        genotypes,
        SEED,
        k=4,
        sampling_methods={
            "random": get_random_sampled_indices,
            "disagreement": get_disagreement_sampled_indices,
            "entropy": get_entropy_sampled_indices,
        },
        run_id=args.run_id,
        plot_dir="plots",
    )

    # Export both LaTeX tables
    export_spearman_table(
        plot_data_df,
        run_id=args.run_id,
        plot_type="sample_for_accuracy",
        output_dir=os.path.join("tables", "spearman"),
        filename_suffix="test_on_accuracy",
    )
    export_spearman_table(
        plot_data_df,
        run_id=args.run_id,
        plot_type="sample_for_accuracy_test_on_length",
        output_dir=os.path.join("tables", "spearman"),
        filename_suffix="test_on_length",
    )

    # save sampled ids to output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    sampled_ids_df = pd.DataFrame(sampled_ids)
    sampled_ids_df.to_csv(args.output_file, index=False)
    print(f"✅ Saved sampled ids: {args.output_file}")


if __name__ == "__main__":
    main()