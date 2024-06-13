# !/bin/bash

# This shell script is used to train and test a model on the MNLI dataset.

# Usage::

#     bash train_and_test_mnli.sh

# Description::

#     This script first prints the current working directory three times.
#     Then, it executes the Python script 'train.py' to train a model on the multinli_1.0 dataset.
#     The trained model is saved to the directory 'MNLI_MODEL'.
#     After training, it executes the Python script 'test.py' to test the trained model on the MNLI and HANS datasets.
#     Paths to the required dataset files are provided as arguments to the test script.

# Example::

#     bash train_and_test_mnli.sh

echo $(pwd)
echo $(pwd)
echo $(pwd)
python3 train.py --dataset_name multinli_1.0 --output_model_directory MNLI_MODEL --output_tokenizer_directory MNLI_MODEL

python3 test.py --input_model_path MNLI_MODEL/BestModel \
                --mnli_train_path ../resources/multinli_1.0/multinli_1.0_train.txt \
                --mnli_val_path ../resources/multinli_1.0/val.pkl \
                --hans_test_path ../resources/HANS/hans1.txt 