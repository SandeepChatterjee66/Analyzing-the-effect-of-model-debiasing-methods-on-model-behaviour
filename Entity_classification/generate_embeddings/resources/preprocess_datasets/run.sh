python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_trng.txt --processed_file_path ../MedMentions/train.txt

python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_dev.txt --processed_file_path ../MedMentions/devel.txt

python3 preprocess.py --dataset datasets/MedMentions/full/data/corpus_pubtator.txt --pmid_file datasets/MedMentions/full/data/corpus_pubtator_pmids_test.txt --processed_file_path ../MedMentions/test.txt


python3 preprocess.py --dataset datasets/BC5CDR/train_corpus.txt --processed_file_path ../BC5CDR/train.txt

python3 preprocess.py --dataset datasets/BC5CDR/dev_corpus.txt --processed_file_path ../BC5CDR/devel.txt

python3 preprocess.py --dataset datasets/BC5CDR/test_corpus.txt --processed_file_path ../BC5CDR/test.txt


python3 preprocess.py --dataset datasets/NCBI_disease/NCBItrainset_corpus.txt --processed_file_path ../NCBI_disease/train.txt

python3 preprocess.py --dataset datasets/NCBI_disease/NCBIdevelopset_corpus.txt --processed_file_path ../NCBI_disease/devel.txt

python3 preprocess.py --dataset datasets/NCBI_disease/NCBItestset_corpus.txt --processed_file_path ../NCBI_disease/test.txt

