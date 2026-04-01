#!/bin/bash
#SBATCH --job-name=extract_samples
#SBATCH --account=IscrC_LENS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/extract_samples_1.5B_math%j.out

cd /leonardo/home/userexternal/miacobel/project_new/mergenetic/experiments/spearman-ranking/

BASE_MODEL=/leonardo_scratch/fast/IscrC_LENS/miacobel/model/Qwen2.5-Math-1.5B
REASONING_MODEL=/leonardo_scratch/fast/IscrC_LENS/miacobel/model/DeepSeek-R1-Distill-Qwen-1.5B

python extract_samples_new.py \
  --output_file outputs/1.5B/math.csv \
  --evaluation_dir outputs/1.5B/math \
  --task math \
  --base_model $BASE_MODEL \
  --reasoning_model $REASONING_MODEL \
  --device cuda \
  --num_genotypes 11 \
  --run_id math_1.5B