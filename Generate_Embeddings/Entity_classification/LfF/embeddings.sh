#!/bin/bash
# Usage: This script generates embeddings for different datasets using a trained model and tokenizer.
# It first trains a model on each dataset specified in the TRAIN_DATA array, saves the trained model and tokenizer,
# and then generates embeddings for each dataset in the TRAIN_DATA and TEST_DATA arrays using the corresponding trained model.
# The embeddings are saved in the specified output directory.
# Purpose: The purpose of this script is to automate the process of generating embeddings for multiple datasets using
# a single trained model and tokenizer. It simplifies the process of running inference on different datasets
# by encapsulating all necessary steps, including model training and inference, in a single script.

echo $(pwd)
declare -a TRAIN_DATA=("MedMentions" "BC5CDR")
declare -a MODEL_DIR=("MedMentions_MODEL" "BC5CDR_MODEL")
declare -a TEST_DATA=("MedMentions" "BC5CDR" "NCBI_disease")
declare -a PRED_DIRS=("MM-MM" "BC5-BC5" "MM-BC5" "MM-NCBI" "BC5-MM" "BC5-NCBI")

export PRED_FOLDER="Predictions"

for TRAIN in ${TRAIN_DATA[@]}
do
    if [[ $TRAIN == "MedMentions" ]]
    then
        export MODEL="${MODEL_DIR[0]}"
    else
        export MODEL="${MODEL_DIR[1]}"
    fi
    echo $MODEL
    mkdir -p ./Embeddings/$TRAIN
    mkdir -p ./Embeddings/$TEST

    python3 generate_embeddings.py \
          --train_dataset_name $TRAIN \
          --test_dataset_name $TRAIN \
          --model_directory $MODEL/BestModel \
          --tokenizer_directory $MODEL/BestModel \
          --output_file Embeddings/$TRAIN \
          --train_file True

    for TEST in ${TEST_DATA[@]}
    do
        echo -e "\n\n\tGenerating embeddings of $TEST dataset using model trained using $TRAIN dataset\n\n"
        
        echo -e "\n\n\t\t\t----GENERATING EMBEDDINGS----\n\n"
        python3 generate_embeddings.py \
          --train_dataset_name $TRAIN \
          --test_dataset_name $TEST \
          --model_directory $MODEL/BestModel \
          --tokenizer_directory $MODEL/BestModel \
          --output_file Embeddings/$TEST \
          --mapping_file Embeddings/mapping.txt \
          --train_file False

        # echo -e "\n\n\t\t\t----SPLITTING EMBEDDINGS----\n\n"

        # python3 split_embeddings.py \
        #         --train_dataset_name $TRAIN \
        #         --test_dataset_name $TEST \
        #         --cui_dictionary_train entity_cui_train.txt \
        #         --cui_dictionary_test entity_cui_test.txt \
        #         --embeddings Embeddings/$TRAIN \
        #         --mapping_file Embeddings/mapping.txt

    done
done
