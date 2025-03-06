#!/bin/bash

# Enable associative arrays
declare -A METHOD_PATHS

# Configuration
# Evaluation model
MODEL_TYPE="gpt"
MODEL_NAME="gpt-4o"
MODEL_PATH=""

# Model to evaluate
MODEL_TO_EVAL="gpt-4o-mini"
# MODEL_TO_EVAL="meta-llama/Meta-Llama-3.1-70B-Instruct"
# MODEL_TO_EVAL="Qwen/Qwen2.5-72B-Instruct"
# MODEL_TO_EVAL="DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"
HYP_SIZE=20

# Define tasks
TASKS=(
    "admission/level_1/base"
    "admission/level_2/depth_2"
    "admission/level_2/distractor_3"
    "admission/level_2/noise_10"
    "admission/level_2/size_5"
)

HYPOGEN_PATH_PREFIX="/home/haokunliu/hypothesis-generation"
RESULTS_DIR="${HYPOGEN_PATH_PREFIX}/results"
METADATA_BASE="${HYPOGEN_PATH_PREFIX}/data"

HYPOGENIC_COMMON_PATH="hypotheses_training_sample_final_seed_42_epoch_0.json"
INIT_COMMON_PATH="hypotheses_training_sample_0_seed_42_epoch_0.json"
# Define methods and their relative paths
METHOD_PATHS=(
    ["HypoGeniC"]="hyp_${HYP_SIZE}/${HYPOGENIC_COMMON_PATH}"
    ["Zero_shot_gen"]="hyp_${HYP_SIZE}_zero_shot/${INIT_COMMON_PATH}"
    ["IO_prompting"]="IO_refinement/hypotheses_training_sample_init_seed_42_epoch_0.json"
    ["IO_refinement"]="IO_refinement/hypotheses_training_sample_final_seed_42_epoch_2.json"
    # Add more method-path pairs as needed
)

# Base paths


# Iterate through tasks and methods
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"
    METADATA_PATH="${METADATA_BASE}/${TASK}/metadata.json"
    
    for METHOD in "${!METHOD_PATHS[@]}"; do
        echo "Evaluating method: $METHOD"
        # Construct full hypotheses path using the mapping
        HYPOTHESES_PATH="${RESULTS_DIR}/${TASK}/${MODEL_TO_EVAL}/${METHOD_PATHS[$METHOD]}"
        LOG_FILE="${TASK}/${MODEL_TO_EVAL}/${METHOD}"
        
        echo "Running evaluation for ${TASK}/${METHOD}"
        CMD="python evaluations/evaluate.py \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --metadata $METADATA_PATH \
            --hypotheses $HYPOTHESES_PATH \
            --log_file $LOG_FILE \
            --hdr"
            
        echo "Executing: $CMD"
        eval $CMD
        
        echo "----------------------------------------"
    done
done
