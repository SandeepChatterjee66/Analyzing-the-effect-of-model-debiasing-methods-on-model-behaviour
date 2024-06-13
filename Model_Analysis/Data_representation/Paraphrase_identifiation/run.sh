#!/bin/bash

# This script performs pairwise comparisons between different models using k-NN.

echo $(pwd)

#change this to correct embedding path
BASE_DIR="../../resources/Embeddings"

declare -a K=(10 20 50 100 200 500 800 1000)
declare -a TRAIN_DATA=("QQP")
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")
declare -a TEST_DATA=("QQP")
declare -a PRED_DIRS=("QQP")

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
                fi
                echo "$MODEL1 vs $MODEL2"
                echo "Running test on $TEST dataset embeddings from $TRAIN trained model "
                python3 jaccard.py  --file1 $BASE_DIR/$MODEL1/$TEST\_train\_embeddings.txt \
                                       --file2 $BASE_DIR/$MODEL2/$TEST\_train\_embeddings.txt
            done
        done
    done
done