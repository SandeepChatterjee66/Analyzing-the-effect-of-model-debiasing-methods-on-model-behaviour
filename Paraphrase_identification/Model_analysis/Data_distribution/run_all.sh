#!/bin/bash

# This script performs a series of operations to evaluate models using k-nearest neighbors (KNN) and generates results for different datasets and models.

# Usage::

#     bash evaluate_models.sh

# Returns::

#     Final output stored in 'final_output.txt' containing the results of model evaluations.

echo $(pwd)

declare -a K=(10 20 50 100 200 500 800 1000)
declare -a TRAIN_DATA=("QQP")
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")
declare -a TEST_DATA=("QQP""PAWS")
declare -a PRED_DIRS=("QQP""PAWS")

for MODEL in ${MODELS[@]}
do
    for TRAIN in ${TRAIN_DATA[@]}
    do
        for TEST in ${TEST_DATA[@]}
        do
            if [[ $TRAIN == "QQP" ]]
            then
                if [[ $TEST == "QQP" ]]
                then
                    export PRED_DIR=${PRED_DIRS[0]}
                elif [[ $TEST == "PAWS" ]]
                then
                    export PRED_DIR=${PRED_DIRS[1]}
                elif [[ $TEST == "PAWS" ]]
                then
                    export PRED_DIR=${PRED_DIRS[2]}
                fi
            fi
            echo $MODEL
            echo $PRED_DIR
            mkdir -p "./results/$MODEL"
            python3 ood.py --input_file_train ../resources/Embeddings/$MODEL/$TRAIN\_train_embeddings.txt \
                                --train_model_name $MODEL \
                                --input_file_val ../resources/Embeddings/$MODEL/$TRAIN\_eval_embeddings.txt \
                                --input_file2 ../resources/Embeddings/$MODEL/$TEST\_embeddings.txt \
                                --pred_dir $PRED_DIR \
                                --test_groundtruth ../resources/Embeddings/paws_groundtruth.txt \
                                --train_groundtruth ../resources/Embeddings/qqp_train_groundtruth.txt \
                                --percent 95
        done
    done
done


bash run_iou.sh > final_output.txt