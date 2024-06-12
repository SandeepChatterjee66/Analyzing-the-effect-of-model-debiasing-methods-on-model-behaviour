#!/bin/bash

# This script calculates jaccard similarity (IOU) evaluate models using k-nearest neighbors (KNN) and generates results for different datasets and models.

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


for ((i=0; i<${#MODELS[@]}; i++))
do
    for ((j=i+1; j<${#MODELS[@]}; j++))
    do
        MODEL1=${MODELS[$i]}
        MODEL2=${MODELS[$j]}
        echo $MODEL1
        echo $MODEL2
        for TRAIN in ${TRAIN_DATA[@]}
        do
            echo TRAIN
            for TEST in ${TEST_DATA[@]}
            do
                echo TEST
                for k_val in ${K[@]}
                do
                    echo k_val
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
                    echo "$MODEL1 vs $MODEL2"
                    echo "Running test on $TEST dataset embeddings from $TRAIN trained model for K = $k_val"
                    python3 evaluate.py --file1 ./results/$MODEL1/k_$k_val/$PRED_DIR/$PRED_DIR\.pkl \
                                        --file2 ./results/$MODEL2/k_$k_val/$PRED_DIR/$PRED_DIR\.pkl
                done
            done
        done
    done
done