#!/bin/bash

# This Bash script is used to compare different models using Jaccard similarity coefficients. It iterates over pairs of models, training data, and test data to compare the performance of different models on different datasets.
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
            echo $TRAIN
            for TEST in ${TEST_DATA[@]}
            do
                echo $TEST
                if [[ $TRAIN == "QQP" ]]
                then
                    if [[ $TEST == "QQP" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[0]}
                    elif [[ $TEST == "PAWS" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[1]}
                    fi
                fi
                echo "$MODEL1 vs $MODEL2"
                echo "Running test on $TEST dataset embeddings from $TRAIN trained model "

                python3 jaccard_coefficients.py \
                        --train_dataset_name $TRAIN \
                        --test_dataset_name $TEST \
                        --model1_pred_file ../resources/Predictions/$MODEL1/$TEST.txt \
                        --model2_pred_file ../resources/Predictions/$MODEL2/$TEST.txt \
                        --groundtruth_file ../resources/groundtruth/$TEST\_test_groundtruth.txt
            done
        done
    done
done