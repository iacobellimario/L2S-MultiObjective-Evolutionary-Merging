#!/bin/bash
#SBATCH --job-name=mergekit
#SBATCH --account=IscrC_LENS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:10:00
#SBATCH --output=logs/mergekit%j.out

python /leonardo/home/userexternal/miacobel/project_new/mergenetic/scripts/mergekit.py \
/leonardo/home/userexternal/miacobel/project_new/mergenetic/experiments/evolutionary-merging-qwen/Mergenetic-TA-1.5B-MATH-entropy/config.yaml \
/leonardo_scratch/fast/IscrC_LENS/miacobel/model/Mergenetic/1.5B/Entropy/04