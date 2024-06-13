#!/bin/bash
# This script performs pairwise comparisons between different models using the Jaccard coefficient.

# Print the current working directory
echo $(pwd)

# Define the values of k to be used
declare -a K=(10 20 50 100 200 500 800 1000)

# Define the datasets for training
declare -a TRAIN_DATA=("MNLI")

# Define the models to be compared
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")

# Define the datasets for testing
declare -a TEST_DATA=("HANS" "MNLI" "SNLI")

# Define the directories for predictions
declare -a PRED_DIRS=("MNLI-m" "MNLI-mm" "HANS")

# Loop through each pair of models
for ((i=0; i<${#MODELS[@]}; i++))
do
    for ((j=i+1; j<${#MODELS[@]}; j++))
    do
        MODEL1=${MODELS[$i]}
        MODEL2=${MODELS[$j]}
        echo $MODEL1
        echo $MODEL2

        # Loop through each training dataset
        for TRAIN in ${TRAIN_DATA[@]}
        do
            echo $TRAIN

            # Loop through each testing dataset
            for TEST in ${TEST_DATA[@]}
            do
                echo $TEST

                # Set the prediction directory based on the combination of training and testing datasets
                if [[ $TRAIN == "MNLI" ]]
                then
                    if [[ $TEST == "MNLI-m" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[0]}
                    elif [[ $TEST == "MNLI-mm" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[1]}
                    elif [[ $TEST == "HANS" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[2]}
                    fi
                fi

                # Perform the pairwise comparison between models using the Jaccard coefficient
                echo "$MODEL1 vs $MODEL2"
                echo "Running test on $TEST dataset embeddings from $TRAIN trained model "

                python3 jaccard_coefficients.py \
                        --train_dataset_name $TRAIN \
                        --test_dataset_name $TEST \
                        --model1_pred_file ../resources/Predictions/$MODEL1/$TEST\_pred.txt \
                        --model2_pred_file ../resources/Predictions/$MODEL2/$TEST\_pred.txt \
                        --groundtruth_file ../resources/groundtruth/$TEST\_test_groundtruth.txt
            done
        done
    done
done