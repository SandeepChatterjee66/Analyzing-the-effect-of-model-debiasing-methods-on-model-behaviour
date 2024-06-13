#!/bin/bash


# This script performs a series of operations to evaluate models using k-nearest neighbors (KNN) and generates results for different datasets and models.

# Usage::

#     bash evaluate_models.sh

# Returns::

#     Final output stored in 'final_output.txt' containing the results of model evaluations.




echo $(pwd)
TASK="NLI"
BASE_DIR="../../resources/Embeddings/$TASK"
RESULTS_DIR="./results"

declare -a K=(10 20 50 100 200 500 800 1000)
declare -a TRAIN_DATA=("MNLI")
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")
declare -a TEST_DATA=("MNLI" "SNLI")
declare -a PRED_DIRS=("MNLI" "SNLI")

for MODEL in ${MODELS[@]}
do
    for TRAIN in ${TRAIN_DATA[@]}
    do
        for TEST in ${TEST_DATA[@]}
        do
            if [[ $TRAIN == "MNLI" ]]
            then
                if [[ $TEST == "MNLI" ]]
                then
                    export PRED_DIR=${PRED_DIRS[0]}
                elif [[ $TEST == "SNLI" ]]
                then
                    export PRED_DIR=${PRED_DIRS[1]}

            fi
            echo $MODEL
            echo $PRED_DIR
            mkdir -p "$RESULTS_DIR/$MODEL"
            python3 ood.py --input_file_train $BASE_DIR/$MODEL/$TRAIN\_train_embeddings.txt \
                                --train_model_name $MODEL \
                                --input_file_val $BASE_DIR/$MODEL/$TRAIN\_eval_embeddings.txt \
                                --input_file2 $BASE_DIR/$MODEL/$TEST\_embeddings.txt \
                                --pred_dir $PRED_DIR \
                                --test_groundtruth $BASE_DIR/$TEST\_groundtruth.txt \
                                --train_groundtruth $BASE_DIR/mnli_train_groundtruth.txt \
                                --percent 95
        done
    done
done


bash run_iou.sh > final_output.txt
# python3 plot-sandeep.py
# python3 plot-cosine-sandeep.py