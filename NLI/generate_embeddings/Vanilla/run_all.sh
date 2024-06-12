# !/bin/bash
echo $(pwd)
echo $(pwd)
echo $(pwd)
python3 train.py --dataset_name multinli_1.0 --output_model_directory MNLI_MODEL --output_tokenizer_directory MNLI_MODEL

python3 test.py --input_model_path MNLI_MODEL/BestModel \
                --mnli_train_path ../resources/multinli_1.0/multinli_1.0_train.txt \
                --mnli_val_path ../resources/multinli_1.0/val.pkl \
                --hans_test_path ../resources/HANS/hans1.txt 
