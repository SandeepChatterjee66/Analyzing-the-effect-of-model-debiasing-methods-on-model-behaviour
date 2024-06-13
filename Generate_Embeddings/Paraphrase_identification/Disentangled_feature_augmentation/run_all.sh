# !/bin/bash

# This shell script performs training and testing of a model for the Multi-Genre Natural Language Inference (MNLI) dataset.

# Usage::

#     bash <path_to_script>

# Description::

#     This script first prints the current working directory.
#     Then, it executes the `train.py` script to train a model on the MNLI dataset.
#     The trained model and tokenizer are saved in the specified output directories.
#     After training, the script executes the `test.py` script to evaluate the trained model on the MNLI validation data and the HANS test data.
#     It provides the paths to the trained model, MNLI training data, MNLI validation data, and HANS test data as arguments to the `test.py` script.

# Example::

#     # Assuming the script is in the same directory as the `train.py` and `test.py` scripts
#     bash run_training_testing.sh

echo $(pwd)
echo $(pwd)
echo $(pwd)

python3 train.py --dataset_name QQP --output_model_directory QQP_MODEL --output_tokenizer_directory QQP_MODEL

python3 test.py --input_model_path QQP_MODEL/BestModel \
                --paws_file_path ../resources/PAWS/test.tsv
