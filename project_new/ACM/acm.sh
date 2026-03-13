#!/bin/bash
#SBATCH --job-name=acm
#SBATCH --account=IscrC_LENS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/ACM-TIES-14B.out

cd /leonardo/home/userexternal/miacobel/project_new/ACM/methods

python acm.py \
  --merged_model "/leonardo_scratch/fast/IscrC_LENS/miacobel/model/baseline/14B/TIES" \
  --pretrained_model_name "/leonardo_scratch/fast/IscrC_LENS/miacobel/model/Qwen2.5-14B" \
  --theta 1.2 \
  --save_path "/leonardo_scratch/fast/IscrC_LENS/miacobel/model/ACM/ACM-TIES-14B" 