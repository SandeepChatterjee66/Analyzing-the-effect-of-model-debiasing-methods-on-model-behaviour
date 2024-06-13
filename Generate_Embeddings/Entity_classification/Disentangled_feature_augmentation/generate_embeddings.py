"""
This script is designed to perform token classification tasks using a pre-trained BioBERT model. It includes functionalities to read datasets, generate embeddings, and save them to text files. The script also supports cross-dataset evaluation and handles different datasets like BC5CDR and MedMentions.

Usage::

    python <path_to_script> --train_dataset_name <train_dataset> --test_dataset_name <test_dataset> --model_directory <model_directory> --tokenizer_directory <tokenizer_directory> --mapping_file <mapping_file> --output_file <output_file> --train_file <train_file>

Arguments::

    --train_dataset_name ( str ) : The name of the training dataset (e.g., 'BC5CDR', 'MedMentions').
    --test_dataset_name ( str ) : The name of the test dataset (e.g., 'BC5CDR', 'MedMentions').
    --model_directory ( str ) : The directory containing the pre-trained model.
    --tokenizer_directory ( str ) : The directory containing the tokenizer.
    --mapping_file ( str ) : The path to the mapping file for cross-dataset evaluation (optional).
    --output_file ( str ) : The path to the output file where embeddings or results will be saved.
    --train_file ( str ) : Indicates whether the input is a training file (use 'True' or 'False').

"""
from multiprocessing import reduction
import pandas as pd
import time
import numpy as np
import csv
import argparse
import math
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from seqeval.metrics import classification_report
from config import Config as config
import os
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from Generate_Embeddings.Entity_classification.Disentangled_feature_augmentation.data_loader import load_data
from Generate_Embeddings.Entity_classification.Disentangled_feature_augmentation.data_loader import load_mapping

# Ignore all warnings
warnings.filterwarnings("ignore")

input_path = './'
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512  # suitable for all datasets
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
num_labels = 0

class MainModel(BertPreTrainedModel):
    def __init__(self, config):
        """Initializes the MainModel with a specified configuration.

        Args:
            config (AutoConfig): Configuration object for the model.
        """
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.model_l = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.model_b = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.fc_l_1 = nn.Linear(1536, 2 * (self.num_labels))
        self.fc_l_2 = nn.Linear(2 * (self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2 * (self.num_labels))
        self.fc_b_2 = nn.Linear(2 * (self.num_labels), self.num_labels)

    def features(self, input_ids, attention_mask):
        """Extracts features from the input ids and attention mask.

        Args:
            input_ids (torch.Tensor): Input tensor containing token ids.
            attention_mask (torch.Tensor): Attention mask for the input ids.

        Returns:
            torch.Tensor: Concatenated feature tensor.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask)
        z_b = self.model_b(input_ids, attention_mask=attention_mask)
        z_l = z_l.last_hidden_state[:, 0, :]
        z_b = z_b.last_hidden_state[:, 0, :]
        z = torch.cat((z_l, z_b), dim=1)
        return z

def save_embeddings_to_text_file(embeddings, output_file_path):
    """Saves embeddings to a specified text file.

    Args:
        embeddings (list): List of embeddings to save.
        output_file_path (str): Path to the output text file.
    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            for i, emb in enumerate(embedding):
                if i != len(embedding) - 1:
                    file.write(f'{emb} ')
                else:
                    file.write(f'{emb}\n')

def generate_embeddings(model, dataloader, device, output_file_path):
    """Generates embeddings using the model and saves them to a file.

    Args:
        model (MainModel): The model used to generate embeddings.
        dataloader (DataLoader): DataLoader providing the input data.
        device (str): Device to run the model on ('cuda' or 'cpu').
        output_file_path (str): Path to the output file where embeddings will be saved.

    Returns:
        list: List of generated embeddings.
    """
    embeddings = []
    total_embeddings = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        with torch.no_grad():
            output = model.features(input_ids=input_ids, attention_mask=mask)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path} : {total_embeddings}')
    return embeddings

def main():
    """Main function to execute the script.

    Parses command-line arguments, loads the data, generates embeddings, 
    and saves the embeddings to specified output files.
    """
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dataset_name', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--model_directory', type=str, required=True)
    parser.add_argument('--tokenizer_directory', type=str, required=True)
    parser.add_argument('--mapping_file', type=str, required=False)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    
    args = parser.parse_args()
    global num_labels
    if args.train_dataset_name == "BC5CDR":
        num_labels = 2
    elif args.train_dataset_name == "MedMentions":
        num_labels = 128
    print(f'Dataset : {args.train_dataset_name}')
    print(f'Number of labels : {num_labels}')
    
    input_model_path = os.path.join(input_path, args.model_directory)
    input_tokenizer_path = os.path.join(input_path, args.tokenizer_directory)
    output_file_path = os.path.join(input_path, args.output_file)
    tokenizer = AutoTokenizer.from_pretrained(input_tokenizer_path)
    config = AutoConfig.from_pretrained(input_model_path, num_labels=num_labels)
    model = MainModel.from_pretrained(input_model_path, config=config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    if args.train_dataset_name == 'BC5CDR' and args.test_dataset_name == 'MedMentions':
        mapping = load_mapping(args.mapping_file)
        id2label_test, _, test_data, _ = load_data(args.test_dataset_name, 'test', tokenizer, cross_eval=True, mapping=mapping)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    else:
        if args.train_file == 'True':
            id2label, label2id, train_data, _ = load_data(args.train_dataset_name, 'train', tokenizer)
            _, _, devel_data, _ = load_data(args.train_dataset_name, 'devel', tokenizer)
            train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
            devel_dataloader = DataLoader(devel_data, shuffle=False, batch_size=BATCH_SIZE)
        else:
            id2label, label2id, test_data, _ = load_data(args.test_dataset_name, 'test', tokenizer)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
    if args.train_file == 'True':
        file_name = args.test_dataset_name + '_train.txt'
        file_name_devel = args.test_dataset_name + '_devel.txt'
        output_file_path = os.path.join(input_path, args.output_file, file_name)
        generate_embeddings(model, train_dataloader, device, output_file_path)
        output_file_path = os.path.join(input_path, args.output_file, file_name_devel)
        generate_embeddings(model, devel_dataloader, device, output_file_path)
    else:
        file_name = args.test_dataset_name + '_test.txt'
        output_file_path = os.path.join(input_path, args.output_file, file_name)
        generate_embeddings(model, test_dataloader, device, output_file_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")

if __name__ == '__main__':
    main()
