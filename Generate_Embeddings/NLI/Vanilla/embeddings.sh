mkdir -p ./Embeddings/HANS
mkdir -p ./Embeddings/MNLI

python3 generate_embeddings.py --input_model_path MNLI_MODEL/BestModel \
                --mnli_train_path ../resources/multinli_1.0/multinli_1.0_train.txt \
                --mnli_val_path ../resources/multinli_1.0/val.pkl \
                --hans_test_path ../resources/HANS/hans1.txt 