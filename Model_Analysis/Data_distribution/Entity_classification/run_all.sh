#!/bin/bash

# Script: run_knn_experiments.sh
# Description: Shell script for running KNN Out-of-Distribution Detection experiments
# Usage: bash run_knn_experiments.sh
# Variables:
#   K: Array containing values of k for KNN
#   TRAIN_DATA: Array containing names of training datasets
#   MODELS: Array containing names of models to compare
#   TEST_DATA: Array containing names of test datasets
#   PRED_DIRS: Array containing names of directories for predictions
# Example:
#   bash run_knn_experiments.sh

echo $(pwd)

TASK="Entity_classification"
BASE_DIR="../../resources/Embeddings/$TASK"
RESULTS_DIR="./results"



declare -a K=(10 20 50 100 200 500 800 1000)
declare -a TRAIN_DATA=("MedMentions")
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")
declare -a TEST_DATA=("MedMentions" "BC5CDR" "NCBI_disease")
declare -a PRED_DIRS=("MM_MM" "BC5_BC5" "MM_BC5" "MM_NCBI" "BC5_MM" "BC5_NCBI")
declare -a TASKS=("Entity_classification""NLI""Paraphrase_identification")

# Loop through each tasks
for TASK in ${TASKS[@]}
do
    # Loop through each model
    for MODEL in ${MODELS[@]}
    do
        # Loop through each training dataset
        for TRAIN in ${TRAIN_DATA[@]}
        do
            # Loop through each test dataset
            for TEST in ${TEST_DATA[@]}
            do
                # Set prediction directory based on training and test dataset
                if [[ $TRAIN == "MedMentions" ]]
                then
                    if [[ $TEST == "MedMentions" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[0]}
                    elif [[ $TEST == "BC5CDR" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[2]}
                    elif [[ $TEST == "NCBI_disease" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[3]}
                    fi
                else
                    if [[ $TEST == "MedMentions" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[4]}
                    elif [[ $TEST == "BC5CDR" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[1]}
                    elif [[ $TEST == "NCBI_disease" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[5]}
                    fi
                fi
                echo $MODEL
                echo $PRED_DIR
                # Create directory for results
                mkdir -p "$RESULTS_DIR/$MODEL"
                # Run KNN Out-of-Distribution Detection experiments
                python3 ood.py --train_dataset_name $TRAIN \
                                    --test_dataset_name $TEST \
                                    --input_file_train $BASE_DIR/$TASK/$MODEL/$TRAIN/$TRAIN\_train.txt \
                                    --train_model_name $TASK/$MODEL \
                                    --input_file_val $BASE_DIR/$TASK/$MODEL/$TRAIN/$TRAIN\_devel.txt \
                                    --input_file2 $BASE_DIR/$TASK/$MODEL/$TRAIN/$TEST\_test.txt \
                                    --pred_dir $PRED_DIR \
                                    --test_groundtruth $BASE_DIR/$TEST\_test_groundtruth.txt \
                                    --train_groundtruth $BASE_DIR/$TRAIN\_train_groundtruth.txt \
                                    --percent 95
            done
        done
    done
done

# Redirect output to final_output.txt
bash run_iou.sh > final_output.txt
