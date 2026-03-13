#!/bin/bash
#SBATCH --job-name=baseline_merge
#SBATCH --account=IscrC_LENS 
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=256GB   
#SBATCH --time=01:00:00
#SBATCH --output=logs/TIES%j.out


DEEPSEEK=/leonardo_scratch/fast/IscrC_LENS/miacobel/model/DeepSeek-R1-Distill-Qwen-14B 
QWEN=/leonardo_scratch/fast/IscrC_LENS/miacobel/model/Qwen2.5-14B

# === Task Arithmetic ===
python /leonardo/home/userexternal/miacobel/project_new/Long-to-Short-via-Model-Merging-main/src/main_merging.py \
  --merge_method ties_merging \
  --base_model $QWEN \
  --models_to_merge $DEEPSEEK,$QWEN \
  --scaling_coefficient 0.5 \
  --param_value_mask_rate 0.2 \
  --output_dir /leonardo_scratch/fast/IscrC_LENS/miacobel/model/baseline/14B/TIES \

