# !/bin/bash

# This script generates embeddings for the MNLI and HANS datasets using a pre-trained model.

# Usage::

#     python3 generate_embeddings.py --input_model_path <model_path> --mnli_train_path <mnli_train_path> --mnli_val_path <mnli_val_path> --hans_test_path <hans_test_path>

# Arguments::

#     --input_model_path ( str ) : The path to the pre-trained model.
#     --mnli_train_path ( str ) : The path to the MNLI training data file.
#     --mnli_val_path ( str ) : The path to the MNLI validation data file.
#     --hans_test_path ( str ) : The path to the HANS test data file.

# Description::

#     This script first creates directories for saving embeddings for the MNLI and HANS datasets.
#     Then, it generates embeddings for the MNLI training and validation data using the specified pre-trained model.
#     Next, it generates embeddings for the HANS test data using the same pre-trained model.
#     The generated embeddings are saved in the respective directories.
 



mkdir -p ./Embeddings/HANS
mkdir -p ./Embeddings/MNLI

python3 generate_embeddings.py --input_model_path MNLI_MODEL/BestModel \
                --mnli_train_path ../resources/multinli_1.0/multinli_1.0_train.txt \
                --mnli_val_path ../resources/multinli_1.0/val.pkl \
                --hans_test_path ../resources/HANS/hans1.txt 