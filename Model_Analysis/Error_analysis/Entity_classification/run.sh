#!/bin/bash
declare -a TRAIN_DATA=("MedMentions")
declare -a MODEL_DIR=("MedMentions_MODEL" "BC5CDR_MODEL")
declare -a TEST_DATA=("MedMentions" "BC5CDR" "NCBI_disease")
declare -a PRED_DIRS=("MM-MM" "BC5-BC5" "MM-BC5" "MM-NCBI" "BC5-MM" "BC5-NCBI")

models=("Vanilla" "FeatureSieve" "LfF")

for ((i=0; i<${#models[@]}; i++)); do
    for ((j=i+1; j<${#models[@]}; j++)); do
        model1=${models[$i]}
        model2=${models[$j]}
        echo $model1
        echo $model2
        for TRAIN in ${TRAIN_DATA[@]}
        do
            if [[ $TRAIN == "MedMentions" ]]
            then
                export MODEL="${MODEL_DIR[0]}"
            else
                export MODEL="${MODEL_DIR[1]}"
            fi
            echo $MODEL

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
                
                
                echo -e "\n\n\t\t\t----TESTING STARTED----\n\n"
                python3 jaccard_coefficients.py \
                        --train_dataset_name $TRAIN \
                        --test_dataset_name $TEST \
                        --all_entities_file $TEST/all_entities.txt \
                        --model1_pred_file ../resources/Predictions/$model1/$PRED_DIR/pred.txt \
                        --model2_pred_file ../resources/Predictions/$model2/$PRED_DIR/pred.txt \
                        --groundtruth_file ../resources/groundtruth/$TEST\_test_groundtruth.txt \
                        --mapping_file mapping.txt

            done
        done
    done
done
