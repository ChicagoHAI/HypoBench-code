#!/bin/bash

# Enable associative arrays
declare -A METHOD_PATHS

# Configuration
MODEL_TYPE="gpt"
MODEL_NAME="gpt-4o-mini"
HYP_SIZE=20

# Define tasks
TASKS=(
    "deceptive_reviews"
    "llamagc_detect"
    "gptgc_detect"
    "persuasive_pairs"
    "dreaddit"
)

# Define methods and their relative paths
METHOD_PATHS=(
    ["hypogenic"]="base_output/hypotheses.json"

    ["IO_refinement"]="io_refine_output/final_hypotheses.json"
    # Add more method-path pairs as needed
)

# Base paths
RESULTS_DIR="results"
METADATA_BASE="data"

# Iterate through tasks and methods
for TASK in "${TASKS[@]}"; do
    echo "Processing task: $TASK"
    METADATA_PATH="${METADATA_BASE}/${TASK}/metadata.json"
    
    for METHOD in "${!METHOD_PATHS[@]}"; do
        echo "Evaluating method: $METHOD"
        # Construct full hypotheses path using the mapping
        HYPOTHESES_PATH="${RESULTS_DIR}/${TASK}/${MODEL_NAME}/${METHOD_PATHS[$METHOD]}"
        
        echo "Running evaluation for ${TASK}/${METHOD}"
        CMD="python evaluations/evaluate.py \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --metadata $METADATA_PATH \
            --hypotheses $HYPOTHESES_PATH \
            --all"
            
        echo "Executing: $CMD"
        eval $CMD
        
        echo "----------------------------------------"
    done
done
