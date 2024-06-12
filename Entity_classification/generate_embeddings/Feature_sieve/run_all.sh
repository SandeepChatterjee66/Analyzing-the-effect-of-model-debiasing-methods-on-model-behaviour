#!/bin/bash
echo $(pwd)
declare -a TRAIN_DATA=("MedMentions")
declare -a MODEL_DIR=("MedMentions_MODEL" "BC5CDR_MODEL")
declare -a TEST_DATA=("MedMentions")
declare -a PRED_DIRS=("MM-MM" "BC5-BC5" "MM-BC5" "MM-NCBI" "BC5-MM" "BC5-NCBI")

export PRED_FOLDER="Predictions"

export check_train
echo PRESS 1 to disable training
echo PRESS 0 to continue with training
read check_train

for TRAIN in ${TRAIN_DATA[@]}
do
    if [[ $TRAIN == "MedMentions" ]]
    then
        export MODEL="${MODEL_DIR[0]}"
    else
        export MODEL="${MODEL_DIR[1]}"
    fi
    echo $MODEL
    mkdir -p "$MODEL"

    if [ $check_train == 0 ]
    then
        echo -e "\n\n\t\t\t----TRAINING STARTED----\n\n"
        python3 train.py \
                --dataset_name $TRAIN \
                --output_model_directory $MODEL \
                --output_tokenizer_directory $MODEL
    elif [ $check_train == 1 ]
    then
        echo -e "\n\n\t\t\t----TRAINING SKIPPED----\n\n"
    else
        echo -e "\n\n\t\t\t----INVALID CHOICE----\n\n"
    fi

    for TEST in ${TEST_DATA[@]}
    do
        echo -e "\n\n\tRunning evaluation of $TEST dataset on model trained using $TRAIN dataset\n\n"
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
        
        echo Output prediction file in folder $PRED_FOLDER/$PRED_DIR
        mkdir -p "$PRED_FOLDER/$PRED_DIR"
        
        echo -e "\n\n\t\t\t----TESTING STARTED----\n\n"
        python3 test.py \
                --train_dataset_name $TRAIN \
                --test_dataset_name $TEST \
                --model_directory $MODEL/BestModel \
                --tokenizer_directory $MODEL/BestModel \
                --output_file $PRED_FOLDER/$TEST/pred.txt \
                --mapping_file $PRED_FOLDER/$PRED_DIR/mapping.txt



    done
done