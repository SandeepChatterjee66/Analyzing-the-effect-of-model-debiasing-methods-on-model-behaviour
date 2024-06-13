mkdir -p ./Embeddings/PAWS/
mkdir -p ./Embeddings/QQP/
python3 generate_embeddings.py  --input_model_path ./QQP_MODEL/BestModel \
                                --train_file_path ./QQP/questions.csv \
                                --dev_file_path ./PAWS/dev.tsv