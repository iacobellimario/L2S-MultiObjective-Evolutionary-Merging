# ==== Imports ====
import os
import json
import torch
import random
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# pymoo components
from mergenetic.merging.taskarithmetic_merger import TaskArithmeticMerger

# Mergekit and Mergenetic
from mergenetic.utils import ConfigLmEval
from mergenetic import PROJECT_ROOT

# Hugging Face 
from huggingface_hub import snapshot_download

method_to_label = {
    'random': 'Random',
    'disagreement': 'Disagreement',
    'entropy': 'Entropy'
}

# ==== Set the seed and max number of evaluated samples ====
SEED = 42
NUM_SAMPLES = 1000

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Compute ranking of merged models with different genotypes and extract samples that maximize entropy.")
    parser.add_argument("--output_file", type=str, default="outputs/extracted_samples.csv", help="Output file to store extracted sample ids.")
    parser.add_argument("--evaluation_dir", type=str, default="outputs", help="Directory to store evaluation results.")
    parser.add_argument("--task", type=str, default="gsm8k", help="Task name for evaluation.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B", help="Base model name.")
    parser.add_argument("--reasoning_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Reasoning model name.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation.")
    parser.add_argument("--num_genotypes", type=int, default=2, help="Number of genotypes to evaluate.")

    return parser.parse_args()

def extract_index_and_score(file_path, bench_name, genotype):
    """
    Reads a JSONL file and extracts 'index' and 'score' from each line.
    
    Args:
        file_path (str): Path to the JSONL file.
    
    Returns:
        list: A list of dictionaries, each containing 'index' and 'score'.
              If a line does not have both keys, it is skipped.
    """
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            #print(f"Reading file: {file_path}")
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data = json.loads(line)
                        if 'idx' in data and 'score' in data:
                            results.append({
                                'index': data['idx'],
                                'score': data['score'][0],
                                'benchmark': bench_name,
                                'genotype': genotype,
                                'answer_length': len(data['code'][0]) if 'code' in data else None
                            })
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        print(f"Skipping invalid JSON line: {line}")
                        continue
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {str(e)}")
    
    return results

def update_accuracies(performance_df, correctness_df, benchmarks, genotypes):
    """Update the 'accuracy' column in performance_df with recalculated values from correctness_df."""
    for benchmark in benchmarks:
        for genotype in genotypes:
            accuracy = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == genotype)]['score'].sum() / len(correctness_df[correctness_df['benchmark'] == benchmark]['index'].unique())
            performance_df.loc[(performance_df['benchmark'] == benchmark) & (performance_df['genotype'] == genotype), 'accuracy'] = accuracy
    # now let's add a new column for mean_answer_length
    for benchmark in benchmarks:
        for genotype in genotypes:
            mean_length = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == genotype)]['answer_length'].mean()
            performance_df.loc[(performance_df['benchmark'] == benchmark) & (performance_df['genotype'] == genotype), 'mean_answer_length'] = mean_length
        # normalize mean_answer_length to [0, 1] per benchmark using the mean_length of genotype 0.0
        max_length = performance_df[(performance_df['benchmark'] == benchmark)]['mean_answer_length'].max()
        min_length = performance_df[(performance_df['benchmark'] == benchmark)]['mean_answer_length'].min()
        performance_df.loc[performance_df['benchmark'] == benchmark, 'mean_answer_length'] = (performance_df[performance_df['benchmark'] == benchmark]['mean_answer_length'] - min_length) / (max_length - min_length) if max_length != min_length else None
    return performance_df

def get_random_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score='score'):
    """Get sampled indices for each benchmark based on the sample percentage."""
    random.seed(seed)
    idxes_per_benchmark = {benchmark: correctness_df[correctness_df['benchmark'] == benchmark]['index'].unique().tolist() for benchmark in benchmarks}
    all_sampled_idx = {}
    for benchmark in benchmarks:
        sample_size = max(1, int(sample_percent * len(idxes_per_benchmark[benchmark])))
        sampled_idxes = random.sample(idxes_per_benchmark[benchmark], sample_size)
        all_sampled_idx[benchmark] = sampled_idxes
    return all_sampled_idx

def get_disagreement_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score='score'):
    """Get sampled indices prioritizing disagreements between genotypes 0.0 and 1.0."""
    random.seed(seed)
    all_sampled_idx = {}
    if not all(g in correctness_df['genotype'].unique() for g in [0.0, 1.0]):
        raise ValueError("Both genotypes 0.0 and 1.0 must be present in correctness_df for disagreement sampling.")
    for benchmark in benchmarks:
        idxes = correctness_df[correctness_df['benchmark'] == benchmark]['index'].unique().tolist()
        # Get scores for genotype 0.0 and 1.0
        scores_0 = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == 0.0)].set_index('index')[score]
        scores_1 = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == 1.0)].set_index('index')[score]
        disagreement_indices = []
        for idx in idxes:
            if idx in scores_0.index and idx in scores_1.index:
                if score == 'answer_length':
                    if (scores_0.loc[idx] - scores_1.loc[idx]) / scores_0.loc[idx] > 0.5:  # consider a threshold for floating point comparison
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
                print(f"Warning: Not enough indices for benchmark {benchmark}, sampled {len(sampled_idxes)} out of {sample_size}")
        all_sampled_idx[benchmark] = sampled_idxes
    return all_sampled_idx

def compute_sample_accuracies(performance_df, correctness_df, benchmarks, genotypes, all_sampled_idx):
    """Compute sample_accuracy in performance_df using sampled indices."""
    perf_df = performance_df.copy()[performance_df['benchmark'].isin(benchmarks) & performance_df['genotype'].isin(genotypes)]
    for benchmark in benchmarks:
        for genotype in genotypes:
            sampled_idxes = all_sampled_idx[benchmark]
            if len(sampled_idxes) == 0:
                continue
            sample_score = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == genotype) & (correctness_df['index'].isin(sampled_idxes))]['score'].sum() / len(sampled_idxes)
            sample_mean_length = correctness_df[(correctness_df['benchmark'] == benchmark) & (correctness_df['genotype'] == genotype) & (correctness_df['index'].isin(sampled_idxes))]['answer_length'].sum() / len(sampled_idxes)
            perf_df.loc[(perf_df['benchmark'] == benchmark) & (perf_df['genotype'] == genotype), 'sample_acc'] = sample_score
            perf_df.loc[(perf_df['benchmark'] == benchmark) & (perf_df['genotype'] == genotype), 'sample_length'] = sample_mean_length
    return perf_df

def compute_spearman_correlations(perf_df, benchmarks, col1='accuracy', col2='sample_acc'):
    """Compute Spearman correlations between 'accuracy' and 'sample_acc' for each benchmark."""
    spearman_correlations = {}
    for benchmark in benchmarks:
        col1_scores = perf_df[perf_df['benchmark'] == benchmark][col1]
        col2_scores = perf_df[perf_df['benchmark'] == benchmark][col2]
        if not col1_scores.empty and not col2_scores.empty:
            if col1_scores.nunique() <= 1 or col2_scores.nunique() <= 1:
                corr = np.nan
            else:
                corr, _ = spearmanr(col1_scores, col2_scores)
            spearman_correlations[benchmark] = corr
    return spearman_correlations

def get_entropy_sampled_indices(correctness_df, benchmarks, seed, sample_percent=0.01, score='score'):
    """Get sampled indices prioritizing high entropy samples across genotypes."""
    random.seed(seed)
    all_sampled_idx = {}
    genotypes = correctness_df['genotype'].unique().tolist()
    sampled_genotypes = random.sample(genotypes, min(len(genotypes)-1, len(genotypes)))

    for benchmark in benchmarks:
        df_b = correctness_df[correctness_df['benchmark'] == benchmark]
        # filter only sampled genotypes
        df_b = df_b[df_b['genotype'].isin(sampled_genotypes)]
        grouped = df_b.groupby('index')[score].agg(['mean', 'count'])
        p1 = grouped['mean']
        p0 = 1 - p1
        # Handle edge cases where p1 or p0 are 0 to avoid log2(0)
        term1 = np.zeros_like(p1)
        term2 = np.zeros_like(p0)
        
        # Only calculate log terms where p1 and p0 are valid (not 0)
        valid_p1 = (p1 > 0)
        valid_p0 = (p0 > 0)
        
        term1[valid_p1] = p1[valid_p1] * np.log2(p1[valid_p1])
        term2[valid_p0] = p0[valid_p0] * np.log2(p0[valid_p0])
        entropy = - (term1 + term2)
        entropy_dict = dict(zip(grouped.index, entropy))
        idxes = df_b['index'].unique()
        for idx in idxes:
            if idx not in entropy_dict:
                entropy_dict[idx] = 0
        # Sort indices by entropy descending
        high_entropy_indices = sorted(idxes, key=lambda x: entropy_dict[x], reverse=True)
        
        sample_size = max(1, int(sample_percent * len(idxes)))
        if len(high_entropy_indices) >= sample_size:
            sampled_idxes = high_entropy_indices[:sample_size]
        else:
            sampled_idxes = high_entropy_indices.copy()
            remaining = [idx for idx in idxes if idx not in high_entropy_indices]
            additional_needed = sample_size - len(high_entropy_indices)
            additional = random.sample(remaining, min(additional_needed, len(remaining)))
            sampled_idxes.extend(additional)
            if len(sampled_idxes) < sample_size:
                print(f"Warning: Not enough indices for benchmark {benchmark}, sampled {len(sampled_idxes)} out of {sample_size}")
        all_sampled_idx[benchmark] = sampled_idxes
    return all_sampled_idx

def plot_spearman_vs_sampling(performance_df, correctness_df, benchmarks, genotypes, val_genotypes, seed, k=5, sampling_methods={'random': get_random_sampled_indices}):
    import numpy as np
    from scipy.stats import t

    sample_percents = np.linspace(0.05, 1.0, 10)
    spearman_results_acc = {method_name: {benchmark: [] for benchmark in benchmarks} for method_name in sampling_methods}
    spearman_results_acc_for_len = {method_name: {benchmark: [] for benchmark in benchmarks} for method_name in sampling_methods}
    spearman_results_len = {method_name: {benchmark: [] for benchmark in benchmarks} for method_name in sampling_methods}

    sampled_ids = []

    for method_name, method_func in sampling_methods.items():
        for sample_percent in sample_percents:
            fold_corrs_acc = {benchmark: [] for benchmark in benchmarks}
            fold_corrs_acc_for_len = {benchmark: [] for benchmark in benchmarks}
            fold_corrs_len = {benchmark: [] for benchmark in benchmarks}
            for fold in range(k):
                fold_seed = seed + fold  # vary seed per fold
                val_correctness_df = correctness_df[correctness_df['genotype'].isin(val_genotypes)]

                sampled_ids_acc = method_func(val_correctness_df, benchmarks, fold_seed, sample_percent)
                    
                sampled_ids_len = method_func(val_correctness_df, benchmarks, fold_seed, sample_percent, score='answer_length')

                for benchmark in benchmarks:
                    sampled_ids.append({
                        'method': method_name,
                        'sample_percent': sample_percent,
                        'fold': fold,
                        'seed': fold_seed,
                        'sampled_ids': sampled_ids_acc[benchmark],
                        'score': 'accuracy',
                        'benchmark': benchmark
                    })
                    sampled_ids.append({
                        'method': method_name,
                        'sample_percent': sample_percent,
                        'fold': fold,
                        'seed': fold_seed,
                        'sampled_ids': sampled_ids_len[benchmark],
                        'score': 'answer_length',
                        'benchmark': benchmark
                    })

                #print(f"Method: {method_name}, Sample Percent: {sample_percent:.2f}, Fold: {fold+1}, Sampled IDs: {sampled_ids}")
                temp_performance_acc_df = compute_sample_accuracies(performance_df.copy(), correctness_df, benchmarks, genotypes, sampled_ids_acc)
                temp_performance_len_df = compute_sample_accuracies(performance_df.copy(), correctness_df, benchmarks, genotypes, sampled_ids_len)
                #print(f"Temp Performance DF (Method: {method_name}, Sample Percent: {sample_percent:.2f}, Fold: {fold+1}):\n{temp_performance_df[['benchmark', 'genotype', 'accuracy', 'sample_accuracy']]}")
                spearman_correlations_acc = compute_spearman_correlations(temp_performance_acc_df, benchmarks)
                spearman_correlations_acc_for_len = compute_spearman_correlations(temp_performance_acc_df, benchmarks, 'mean_answer_length', 'sample_length')
                spearman_correlations_len = compute_spearman_correlations(temp_performance_len_df, benchmarks, 'mean_answer_length', 'sample_length')

                for benchmark in benchmarks:
                    fold_corrs_acc[benchmark].append(spearman_correlations_acc.get(benchmark, np.nan))
                    fold_corrs_acc_for_len[benchmark].append(spearman_correlations_acc_for_len.get(benchmark, np.nan))
                    fold_corrs_len[benchmark].append(spearman_correlations_len.get(benchmark, np.nan))
            for benchmark in benchmarks:
                corrs_acc = np.array(fold_corrs_acc[benchmark])
                valid_corrs_acc = corrs_acc[~np.isnan(corrs_acc)]
                corrs_acc_for_len = np.array(fold_corrs_acc_for_len[benchmark])
                valid_corrs_acc_for_len = corrs_acc_for_len[~np.isnan(corrs_acc_for_len)]
                corrs_len = np.array(fold_corrs_len[benchmark])
                valid_corrs_len = corrs_len[~np.isnan(corrs_len)]
                if len(valid_corrs_acc) == 0:
                    mean_corr_acc = np.nan
                    ci_acc = np.nan
                    mean_corr_acc_for_len = np.nan
                    ci_acc_for_len = np.nan
                    mean_corr_len = np.nan
                    ci_len = np.nan
                elif len(valid_corrs_acc) == 1:
                    mean_corr_acc = valid_corrs_acc[0]
                    ci_acc = 0  # No CI for single value
                    mean_corr_acc_for_len = valid_corrs_acc_for_len[0]
                    ci_acc_for_len = 0  # No CI for single value
                    mean_corr_len = valid_corrs_len[0]
                    ci_len = 0  # No CI for single value
                else:
                    mean_corr_acc = np.mean(valid_corrs_acc)
                    std_corr_acc = np.std(valid_corrs_acc, ddof=1)
                    ci_acc = t.ppf(0.975, len(valid_corrs_acc)-1) * std_corr_acc / np.sqrt(len(valid_corrs_acc))

                    mean_corr_acc_for_len = np.mean(valid_corrs_acc_for_len)
                    std_corr_acc_for_len = np.std(valid_corrs_acc_for_len, ddof=1)
                    ci_acc_for_len = t.ppf(0.975, len(valid_corrs_acc_for_len)-1) * std_corr_acc_for_len / np.sqrt(len(valid_corrs_acc_for_len))

                    mean_corr_len = np.mean(valid_corrs_len)
                    std_corr_len = np.std(valid_corrs_len, ddof=1)
                    ci_len = t.ppf(0.975, len(valid_corrs_len)-1) * std_corr_len / np.sqrt(len(valid_corrs_len))
                spearman_results_len[method_name][benchmark].append((mean_corr_len, ci_len))
                spearman_results_acc_for_len[method_name][benchmark].append((mean_corr_acc_for_len, ci_acc_for_len))
                spearman_results_acc[method_name][benchmark].append((mean_corr_acc, ci_acc))
    
    # Create plot data dataframe
    plot_data = []
    
    # Create subplots: 2 columns for each benchmark, one for accuracy, one for length
    fig, axes = plt.subplots(nrows=len(benchmarks), ncols=3, figsize=(12, len(benchmarks) * 4))
    for i, benchmark in enumerate(benchmarks):
      if len(benchmarks) == 1:
        ax_acc = axes[0]
        ax_acc_for_len = axes[1]
        ax_len = axes[2]
      else:
        ax_acc = axes[i, 0]
        ax_acc_for_len = axes[i, 1]
        ax_len = axes[i, 2]
      for method_name in sampling_methods:
        means_acc = [spearman_results_acc[method_name][benchmark][j][0] for j in range(len(sample_percents))]
        cis_acc = [spearman_results_acc[method_name][benchmark][j][1] for j in range(len(sample_percents))]

        means_acc_for_len = [spearman_results_acc_for_len[method_name][benchmark][j][0] for j in range(len(sample_percents))]
        cis_acc_for_len = [spearman_results_acc_for_len[method_name][benchmark][j][1] for j in range(len(sample_percents))]

        means_len = [spearman_results_len[method_name][benchmark][j][0] for j in range(len(sample_percents))]
        cis_len = [spearman_results_len[method_name][benchmark][j][1] for j in range(len(sample_percents))]

        # Store plot data
        for j, sample_percent in enumerate(sample_percents):
          plot_data.append({
            'benchmark': benchmark,
            'method': method_name,
            'sample_percent': sample_percent,
            'plot_type': 'sample_for_accuracy',
            'mean_correlation': means_acc[j],
            'ci_correlation': cis_acc[j]
          })
          plot_data.append({
            'benchmark': benchmark,
            'method': method_name,
            'sample_percent': sample_percent,
            'plot_type': 'sample_for_accuracy_test_on_length',
            'mean_correlation': means_acc_for_len[j],
            'ci_correlation': cis_acc_for_len[j]
          })
          plot_data.append({
            'benchmark': benchmark,
            'method': method_name,
            'sample_percent': sample_percent,
            'plot_type': 'sample_for_length',
            'mean_correlation': means_len[j],
            'ci_correlation': cis_len[j]
          })

        ax_acc.errorbar(sample_percents*100, means_acc, yerr=cis_acc, label=method_to_label[method_name], capsize=5, marker='o')

        ax_acc_for_len.errorbar(sample_percents*100, means_acc_for_len, yerr=cis_acc_for_len, label=method_to_label[method_name], capsize=5, marker='o')

        ax_len.errorbar(sample_percents*100, means_len, yerr=cis_len, label=method_to_label[method_name], capsize=5, marker='o')
      #ax_acc.set_title(f"Sample for Accuracy - {benchmark}")
      ax_acc.set_title(f"Sample for Accuracy")
      #ax_acc.set_xlabel("Sample Percentage (\%)", fontsize=14)
      ax_acc.set_ylabel("Spearman Correlation", fontsize=14)
      ax_acc.set_ylim(-0.1, 1.1)
      ax_acc.axhline(0, color='gray', linestyle='--', linewidth=0.7)
      ax_acc.legend(fontsize=12, loc='lower right')

      ax_acc_for_len.set_title(f"Sample for Accuracy, Test on Answer Length")
      ax_acc_for_len.set_xlabel("Sample Percentage (%)", fontsize=14)
      #ax_acc_for_len.set_ylabel("Spearman Correlation", fontsize=14)
      ax_acc_for_len.set_ylim(-0.1, 1.1)
      ax_acc_for_len.axhline(0, color='gray', linestyle='--', linewidth=0.7)
      ax_acc_for_len.legend(fontsize=12, loc='lower right')

      ax_len.set_title(f"Sample for Answer Length")
      #ax_len.set_xlabel("Sample Percentage (\\%)", fontsize=14)
      #ax_len.set_ylabel("Spearman Correlation", fontsize=14)
      ax_len.set_ylim(-0.1, 1.1)
      ax_len.axhline(0, color='gray', linestyle='--', linewidth=0.7)
      ax_len.legend(fontsize=12, loc='lower right')

    # save to plots/spearman_correlation_vs_sampling.png
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/spearman_correlation_vs_sampling.pdf", bbox_inches='tight')
    plt.close()

    # Convert plot data to DataFrame
    plot_data_df = pd.DataFrame(plot_data)

    return spearman_results_acc, spearman_results_len, sampled_ids, plot_data_df

def main():
    args = parse_args()

    if args.num_genotypes < 2:
        raise ValueError("num_genotypes must be at least 2 to perform evaluation.")

    # Python
    random.seed(SEED)
    # NumPy
    np.random.seed(SEED)
    # PyTorch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(SEED)

    print(f"All seeds set to {SEED}")

    # directory where to store base and merged models
    #model_dir = "/mnt/ssd2/models"
    model_dir = "/leonardo_scratch/fast/IscrC_M4R/miacobel/model"
    os.makedirs(model_dir, exist_ok=True)

    # DeepSeek-R1-Distill-Qwen-1.5B (Base model)
    reasoning_repo = args.reasoning_model
    """
    snapshot_download(
        repo_id=reasoning_repo,
        local_dir=os.path.join(model_dir, args.reasoning_model.split("/")[-1])
    )
    """

    # Qwen2.5-Math-1.5B (Target model)
    qwen_math_repo = args.base_model
    """
    snapshot_download(
        repo_id=qwen_math_repo,
        local_dir=os.path.join(model_dir, args.base_model.split("/")[-1])
    )
    """

    config = ConfigLmEval()

    # Set the absolute path for custom templates
    config.additional_templates_folder = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", "lm_tasks")
    config.bench = args.task
    config.path_to_store_config = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", "experiments", "spearman-ranking", "configs")

    # Reproducibility
    config.seed = SEED  

    # Device for evaluation ("cuda" if available, else "cpu")
    config.device = args.device

    # Run identifier (used for logs, checkpoints, results)
    config.run_id = "Mergenetic-TA-14B-math"

    # Metric to evaluate correctness
    config.metric = "exact_match"

    # Task type (here: focused on math reasoning, e.g. GSM8K)
    config.task_type = "FG_MATH"

    # Paths for saving configs, logs, and merged models
    config.path_to_store_merged_model = f"{model_dir}/merged"

    # Non-canonical setup:
    #   - Deepseek-R1-Distill-Qwen → distillation of DeepSeek-R1 into a Qwen architecture
    #   - Qwen2.5-Math-1.5B        → fine-tuned Qwen2.5 on math reasoning
    config.base_model = f"{model_dir}/{args.reasoning_model.split('/')[-1]}"
    config.models = {"en": f"{model_dir}/{args.base_model.split('/')[-1]}"}

    # Batch size for evaluation
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

    # for each genotype in genotypes, create a merged model and evaluate it
    for index, row in enumerate(genotypes):
        genotype = [row,]
        print(f"Processing genotype {genotype} ({index+1}/{len(genotypes)})")

        # check if results already exist
        if os.path.exists(os.path.join(PROJECT_ROOT, "project_new", "mergenetic", args.evaluation_dir, str(genotype[0]), f"{args.task}", "train_deepseek-math_1000_seed42_t0_s0_e-1_deepseek-math_metrics.json")):
            print(f"Results for genotype {str(genotype[0])} already exist. Skipping evaluation.")
            continue
        print(f"Evaluation not found at {os.path.join(PROJECT_ROOT, 'mergenetic', args.evaluation_dir, str(genotype[0]), f'{args.task}', 'train_deepseek-math_1000_seed42_t0_s0_e-1_deepseek-math_metrics.json')}. Proceeding with evaluation.")
        
        # create a merged model
        merged_model_conf_path = merger.create_individual_configuration(genotype)
        merged_model_path = merger.merge_model_from_configuration(merged_model_conf_path)
        print(f"✅ Merged model created at {merged_model_path}")
        
        # put "CUDA_VISIBLE_DEVICES=1" into env
        subprocess.run(["python3", "../Qwen2.5-Math/evaluation/math_eval.py",
            "--model_name_or_path", f"{merged_model_path}",
            "--data_names", f"{args.task}",
            "--output_dir", f"{args.evaluation_dir}/{genotype[0]}",
            "--prompt_type", "deepseek-math",
            "--n_sampling", "1",
            "--max_tokens_per_call", "10240",
            "--num_shots", "0",
            "--seed", f"{SEED}",
            "--split", "train",
            "--num_test_sample", f"{NUM_SAMPLES}",
            "--use_vllm",
            "--save_outputs",
            "--overwrite",
            "--data_dir", "../Qwen2.5-Math/evaluation/data"
        ], env={**os.environ}, cwd=f"{PROJECT_ROOT}/project_new/mergenetic")

    # After evaluating all genotypes, extract samples that maximize entropy
    results = [] # list of dicts with keys: genotype, benchmark, accuracy
    for genotype in genotypes:
        results_file = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", args.evaluation_dir, str(genotype), f"{args.task}", "train_deepseek-math_1000_seed42_t0_s0_e-1_deepseek-math_metrics.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
                accuracy = data.get("acc", None)
                if accuracy is not None:
                    results.append({
                        "genotype": genotype,
                        "benchmark": args.task,
                        "accuracy": accuracy
                    })
                else :
                    print(f"No accuracy found in {results_file}")
        else:
            print(f"Results file {results_file} does not exist.")

    performance_df = pd.DataFrame(results)

    results_correctness = [] # list of dicts with keys: genotype, benchmark, accuracy
    for genotype in genotypes:
        results_file = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", args.evaluation_dir, str(genotype), f"{args.task}", f"train_deepseek-math_1000_seed{SEED}_t0_s0_e-1.jsonl")
        correctness = extract_index_and_score(results_file, args.task, genotype)
        print(f"Extracted {len(correctness)} correctness entries from {results_file}")
        results_correctness.extend(correctness)

    # Let's add length as additional benchmark metric
    correctness_df = pd.DataFrame(results_correctness)

    # normalize answer length for each benchmark and genotype with min-max normalization
    correctness_df_norm = correctness_df.copy()
    for genotype in genotypes:
        subset = correctness_df_norm[correctness_df_norm['genotype'] == genotype]
        if not subset.empty:
            min_length = subset['answer_length'].min()
            max_length = subset['answer_length'].max()
            if max_length != min_length:
                correctness_df_norm.loc[correctness_df_norm['genotype'] == genotype, 'answer_length'] = (subset['answer_length'] - min_length) / (max_length - min_length)
            else:
                correctness_df_norm.loc[correctness_df_norm['genotype'] == genotype, 'answer_length'] = 0.0  # or some constant value since all lengths are the same

    # update performance_df 'accuracy' column with recalculated accuracies from correctness_df
    performance_df = update_accuracies(performance_df, correctness_df, [args.task], genotypes)

    sp_results_acc, sp_results_len, sampled_ids, plot_data_df = plot_spearman_vs_sampling(performance_df, correctness_df, [args.task], genotypes, genotypes, SEED, k=4, sampling_methods={'random': get_random_sampled_indices, 'disagreement': get_disagreement_sampled_indices, 'entropy': get_entropy_sampled_indices})

    # save sampled ids to output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    sampled_ids_df = pd.DataFrame(sampled_ids)
    sampled_ids_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()