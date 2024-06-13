
# This script is designed to generate embeddings from a pretrained model and save these embeddings to text files.
# It creates directories for storing the embeddings if they do not already exist.

# Usage:

#     bash embeddings.sh

# Arguments:

#     --input_model_path (str): Path to the pretrained model directory.
#     --train_file_path (str): Path to the training dataset file.
#     --dev_file_path (str): Path to the development/validation dataset file.


mkdir -p ../resources/Embeddings/PAWS/
mkdir -p ../resources/Embeddings/QQP/

python3 generate_embeddings.py  --input_model_path ./QQP_MODEL/BestModel \
                                --train_file_path ../resources/QQP/questions.csv \
                                --dev_file_path ../resources/PAWS/dev.tsv