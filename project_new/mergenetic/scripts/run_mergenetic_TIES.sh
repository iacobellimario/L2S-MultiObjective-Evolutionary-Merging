#!/bin/bash
#SBATCH --job-name=mergenetic_qwen_merge
#SBATCH --account=IscrC_LENS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=128GB   
#SBATCH --time=24:00:00
#SBATCH --output=logs/Mergenetic-TIES-1.5B-MATH-Entropy.out

export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn    

python /leonardo/home/userexternal/miacobel/project_new/mergenetic/scripts/Mergenetic-TIES.py
