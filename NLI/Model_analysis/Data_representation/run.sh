#!/bin/bash
# This script performs pairwise comparisons between different models using k-NN.

# Print the current working directory
echo $(pwd)

# Define the values of k to be used
declare -a K=(10 20 50 100 200 500 800 1000)

# Define the datasets for training
declare -a TRAIN_DATA=("MNLI")

# Define the models to be compared
declare -a MODELS=("Vanilla" "Feature_sieve" "DisEnt" "LfF")

# Define the datasets for testing
declare -a TEST_DATA=("MNLI")

# Define the directories for predictions
declare -a PRED_DIRS=("MNLI")

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

                # Set the prediction directory based on the training and testing datasets
                if [[ $TRAIN == "MNLI" ]]
                then
                    if [[ $TEST == "MNLI" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[0]}
                    elif [[ $TEST == "SNLI" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[1]}
                    elif [[ $TEST == "HANS" ]]
                    then
                        export PRED_DIR=${PRED_DIRS[2]}
                    fi
                fi

                # Perform the pairwise comparison between models
                echo "$MODEL1 vs $MODEL2"
                echo "Running test on $TEST dataset embeddings from $TRAIN trained model "
                python3 jaccard.py --file1 ../resources/Embeddings/$MODEL1/$TEST\_train\_embeddings.txt \
                                      --file2 ../resources/Embeddings/$MODEL1/$TEST\_train\_embeddings.txt
            done
        done
    done
done
