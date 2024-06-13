# !/bin/bash
echo $(pwd)
echo $(pwd)
echo $(pwd)

python3 train.py --dataset_name QQP --output_model_directory QQP_MODEL --output_tokenizer_directory QQP_MODEL

python3 test.py --input_model_path QQP_MODEL/BestModel \
                --paws_file_path ../resources/PAWS/test.tsv
