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

RESULTS_DIR="./results"

declare -a K=(10 20 50 100 200 500 800 1000)
declare -a TRAIN_DATA=("MedMentions")
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")
declare -a TEST_DATA=("MedMentions" "BC5CDR" "NCBI_disease")
declare -a PRED_DIRS=("MM_MM" "BC5_BC5" "MM_BC5" "MM_NCBI" "BC5_MM" "BC5_NCBI")


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
                    echo "$MODEL1 vs $MODEL2"
                    echo "Running test on $TEST dataset embeddings from $TRAIN trained model for K = $k_val"
                    python3 evaluate.py --file1 $RESULTS_DIR/$MODEL1/k_$k_val/$PRED_DIR/$PRED_DIR\.pkl \
                                        --file2 $RESULTS_DIR/$MODEL2/k_$k_val/$PRED_DIR/$PRED_DIR\.pkl
                done
            done
        done
    done
done
