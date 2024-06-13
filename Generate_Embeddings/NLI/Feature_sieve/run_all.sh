# !/bin/bash


# This shell script is used to train and test a model for the MNLI dataset.

# Usage::

#     bash train_and_test_mnli.sh

# Description::

#     This script first prints the current working directory three times.
#     Then, it executes the training script 'train.py' to train a model on the MNLI dataset.
#     After training, it executes the testing script 'test.py' to evaluate the trained model on the MNLI validation data and the HANS test data.

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
