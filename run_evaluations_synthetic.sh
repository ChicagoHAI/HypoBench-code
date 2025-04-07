#!/bin/bash

# Enable associative arrays
declare -A METHOD_PATHS

# Configuration
# Evaluation model
MODEL_TYPE="gpt"
MODEL_NAME="gpt-4o"
# MODEL_NAME="gpt-4o-mini"
MODEL_PATH=""

# Model to evaluate
# MODEL_TO_EVAL="gpt-4o-mini"
# MODEL_TO_EVAL="meta-llama/Meta-Llama-3.1-70B-Instruct"
# MODEL_TO_EVAL="Qwen/Qwen2.5-72B-Instruct"
MODEL_TO_EVAL="DeepSeek/DeepSeek-R1-Distill-Llama-70B-local"
HYP_SIZE=20

# Define tasks
TASKS=(
    # "admission/level_1/base"
    # "admission/level_2/depth_2"
    # "admission/level_2/distractor_3"
    # "admission/level_2/noise_10"
    # "admission/level_2/size_5"
    # "shoe"
    # "shoe_two_level/simple"
    # "shoe_two_level/hard"
    # "election/level0"
    # "preference/level0"
    # "election/level1"
    # "preference/level1"
    # "election/level2"
    # "preference/level2"
    # "election/level3"
    # "preference/level3"
    # "election/level4"
    # "preference/level4"
    # "election/level5"
    # "preference/level5"
    # "election/level0_nosubtlety"
    # "preference/level0_nosubtlety"
    # "election/level1_nosubtlety"
    # "preference/level1_nosubtlety"
    # "election/level2_nosubtlety"
    # "preference/level2_nosubtlety"
    # "election/level3_nosubtlety"
    # "preference/level3_nosubtlety"
    # "election/level4_nosubtlety"
    # "preference/level4_nosubtlety"
    # "election/level5_nosubtlety"
    # "preference/level5_nosubtlety"
    # 'election_controlled/5_0_0'
    # 'election_controlled/10_0_0'
    # 'election_controlled/15_0_0'
    # 'election_controlled/20_0_0'
    # 'election_controlled/20_0.1_0'
    # 'election_controlled/20_0.2_0'
    # 'election_controlled/20_0.3_0'
    # 'election_controlled/20_0_0.1'
    # 'election_controlled/20_0_0.2'
    # 'election_controlled/20_0_0.3'
    # 'election_controlled/20_0.1_0.1'
    # 'election_controlled/20_0.2_0.2'
    # 'election_controlled/20_0.3_0.3'
    # 'preference_controlled/5_0_0'
    # 'preference_controlled/10_0_0'
    # 'preference_controlled/15_0_0'
    # 'preference_controlled/20_0_0'
    # 'preference_controlled/20_0.1_0'
    # 'preference_controlled/20_0.2_0'
    # 'preference_controlled/20_0.3_0'
    # 'preference_controlled/20_0_0.1'
    # 'preference_controlled/20_0_0.2'
    # 'preference_controlled/20_0_0.3'
    # 'preference_controlled/20_0.1_0.1'
    # 'preference_controlled/20_0.2_0.2'
    # 'preference_controlled/20_0.3_0.3'
    # 'admission/level_3/depth_3'
    # 'admission/level_3/distractor_6'
    # 'admission/level_3/noise_20'
    # 'admission/level_3/size_10'
    # 'admission/level_4/depth_4'
    # 'admission/level_4/distractor_10'
    # 'admission/level_4/noise_30'
    # 'admission/level_4/size_15'
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