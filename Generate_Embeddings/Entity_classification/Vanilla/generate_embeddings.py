"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --train_dataset_name <train_dataset_name> --test_dataset_name <test_dataset_name> --model_directory <trained_model_directory> --tokenizer_directory <saved_tokenizer_directory> --output_file <path_to_output_pred_file> --mapping_file <mapping_file>

Arguments::

    --train_dataset_name (str): Name of train dataset.
    --test_dataset_name (str): Name of test dataset.
    --model_directory (str): Path to the pretrained model directory.
    --tokenizer_directory (str): Path to the saved model directory.
    --output_file (str): Path to the output prediction file.
    --mapping_file (str): Path to the mapping file.
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
from Generate_Embeddings.Entity_classification.Vanilla.data_loader import load_data, load_mapping

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512# suitable for all datasets
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
num_labels = 0

class MainModel(BertPreTrainedModel):
    """
    Main model class
    """
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config (transformers.PretrainedConfig): Configuration for the model.
            loss_fn (optional): Loss function to use. Defaults to None.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1",config = config)
        self.hidden_layer = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels),self.num_labels)

    def forward(self, input_ids, attention_mask, labels,device):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor): Input IDs of a batch.
            attention_mask (torch.Tensor): Attention mask of a batch.
            labels (torch.Tensor): Target labels of a batch.
            device (str): Device to run the model on(gpu or cpu).
        
        Returns:
            torch.Tensor: Embeddings corresponding to the [CLS] token of the final layer
        """
        output = self.bert(input_ids, attention_mask = attention_mask)
        output = output.last_hidden_state
        output = output[:,0,:]
        return output
    
def save_embeddings_to_text_file(embeddings, output_file_path):
    """
    Save embeddings to a text file. Appends the embeddings to the existing file.
    
    Args:
        embeddings (list): List of embeddings for each batch.
        output_file_path (str): Path to the output file.
        
    Returns:
        None
    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            for i,emb in enumerate(embedding):
                if i != len(embedding) - 1:
                    file.write(f'{emb} ')
                else:
                    file.write(f'{emb}\n')


def generate_embeddings(model, dataloader, device, output_file_path):
    """
    Generate and save embeddings using the model.
    
    Args:
        model (MainModel): The trained model.
        dataloader (DataLoader): DataLoader for the data.
        device (str): Device to use ('cuda' or 'cpu').
        output_file_path (str): Path to the output file.
    
    Returns:
        list: List of embeddings.
    """
    embeddings = []
    total_embeddings = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, labels=targets, device = device)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path} : {total_embeddings}')
    return embeddings

def main():
    """
    Main function to execute the embedding generation.
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
    config = AutoConfig.from_pretrained(input_model_path , num_labels=num_labels)
    model = MainModel.from_pretrained(input_model_path, config = config, loss_fn = None)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    if (args.train_dataset_name == 'BC5CDR' and args.test_dataset_name == 'MedMentions'):
        mapping = load_mapping(args.mapping_file)
        id2label_test,_, test_data,_ = load_data(args.test_dataset_name, 'test', tokenizer, cross_eval=True, mapping = mapping)
        test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
    else:
        if(args.train_file == 'True'):
            id2label, label2id, train_data, _ = load_data(args.train_dataset_name, 'train', tokenizer)
            _,_, devel_data, _ = load_data(args.train_dataset_name, 'devel', tokenizer)
            train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
            devel_dataloader = DataLoader(devel_data, shuffle = False, batch_size=BATCH_SIZE)
        else:
            id2label, label2id, test_data, _ = load_data(args.test_dataset_name, 'test', tokenizer)
            test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
    if(args.train_file == 'True'):
        file_name = args.test_dataset_name + '_train.txt'
        file_name_devel = args.test_dataset_name + '_devel.txt'
        output_file_path = os.path.join(input_path, args.output_file , file_name)
        generate_embeddings(model, train_dataloader, device, output_file_path)
        output_file_path = os.path.join(input_path, args.output_file , file_name_devel)
        generate_embeddings(model, devel_dataloader, device, output_file_path)
    else:
        file_name = args.test_dataset_name + '_test.txt'
        output_file_path = os.path.join(input_path, args.output_file , file_name)
        generate_embeddings(model, test_dataloader, device, output_file_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()