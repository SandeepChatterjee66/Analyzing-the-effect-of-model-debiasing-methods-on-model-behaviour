#!/bin/bash

#!/bin/bash
# This script performs pairwise comparisons between different models using k-NN.


echo $(pwd)

#change this to correct embedding path
BASE_DIR="../../resources/Embeddings"

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
            echo $TRAIN
            for TEST in ${TEST_DATA[@]}
            do
                echo $TEST
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
                echo "Running test on $TEST dataset embeddings from $TRAIN trained model "
                python3 jaccard.py --file1 $BASE_DIR/Embeddings/$MODEL1/$TRAIN/$TEST\_test.txt \
                                    --file2 $BASE_DIR/Embeddings/$MODEL2/$TRAIN/$TEST\_test.txt
            done
        done
    done
done

## have to change - pending
# test.txt should be changed to training as we are taking test embedding