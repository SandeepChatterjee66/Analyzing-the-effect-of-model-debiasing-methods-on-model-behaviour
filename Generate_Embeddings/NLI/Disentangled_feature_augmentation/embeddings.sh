# This script generates embeddings for the MNLI and HANS datasets using a pre-trained model and saves them to the specified directory.

# Usage::

#     python3 generate_embeddings.py --input_model_path <model_path> --mnli_train_path <mnli_train_path> --mnli_val_path <mnli_val_path> --hans_test_path <hans_test_path>

# Arguments::

#     --input_model_path ( str ) : The path to the directory containing the pre-trained model.
#     --mnli_train_path ( str ) : The path to the MNLI training data file.
#     --mnli_val_path ( str ) : The path to the pickled MNLI validation data.
#     --hans_test_path ( str ) : The path to the HANS test data file.

# Description::

#     This script generates embeddings for the MNLI (Multi-Genre Natural Language Inference) and HANS (Heuristic Analysis for NLI Systems) datasets using a pre-trained model.
#     It first creates directories for storing the generated embeddings for MNLI and HANS datasets.
#     Then, it loads the pre-trained model from the specified input_model_path.
#     Next, it loads the MNLI training data, MNLI validation data, and HANS test data.
#     The script generates embeddings for the MNLI training and validation data, as well as for the HANS test data using the loaded model.
#     Finally, it saves the generated embeddings to the respective directories.




mkdir -p ./Embeddings/HANS
mkdir -p ./Embeddings/MNLI

python3 generate_embeddings.py --input_model_path MNLI_MODEL/BestModel \
                --mnli_train_path ../resources/multinli_1.0/multinli_1.0_train.txt \
                --mnli_val_path ../resources/multinli_1.0/val.pkl \
                --hans_test_path ../resources/HANS/hans1.txt 