# !/bin/bash

# This shell script is used to generate embeddings using a pre-trained model for the MNLI dataset.

# Usage::

#     bash generate_mnli_embeddings.sh

# Description::

#     This script first prints the current working directory three times.
#     Then, it executes the Python script 'generate_embeddings.py' to generate embeddings using a pre-trained model for the MNLI dataset.
#     The generated embeddings are saved to a file named 'live-LfF-embedding.txt'.
#     The '&' symbol at the end of the command runs the script in the background.

# Example::

#     bash generate_mnli_embeddings.sh


echo $(pwd)
echo $(pwd)
echo $(pwd)

python3 -u generate_embeddings.py --input_model_path MNLI_MODEL/BestModel > live-LfF-embedding.txt &

