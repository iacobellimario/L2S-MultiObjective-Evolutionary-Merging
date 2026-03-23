#!/bin/bash
#SBATCH --job-name=math_eval      
#SBATCH --account=IscrC_LENS 
#SBATCH --partition=boost_usr_prod  
#SBATCH --output=logs/baseline/deepseek-r1-1.5B.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=128GB             
#SBATCH --time=24:00:00
        
cd /leonardo/home/userexternal/miacobel/project_new/Qwen2.5-Math/evaluation

export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn         
export CUDA_VISIBLE_DEVICES="0"

set -ex

PROMPT_TYPE="deepseek-math"
MAX_TOKEN=10240
NUM_SHOTS=0
DATA_NAME="aime24,gsm8k,college_math,math500,minerva_math,olympiadbench"
MODEL_PATH="/leonardo_scratch/fast/IscrC_LENS/miacobel/model/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="outputs/baseline/DeepSeek-R1-Distill-Qwen-1.5B"
SEED=0
SPLIT="test"
NUM_TEST_SAMPLE=-1

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature 0 \
    --n_sampling 1 \
    --max_tokens_per_call ${MAX_TOKEN} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --num_shots ${NUM_SHOTS} \
    --overwrite \
    --save_outputs 
