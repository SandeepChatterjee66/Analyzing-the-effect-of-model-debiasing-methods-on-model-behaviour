"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> --train_file_path <path_to_train_data> --dev_file_path <path_to_dev_data>

Arguments::

    --input_model_path (str): Path to the pretrained BERT model.
    --train_file_path (str): Path to the training data file.
    --dev_file_path (str): Path to the development data file.

"""
from multiprocessing import reduction
import pandas as pd
import gc
import time
import numpy as np
import csv
import argparse
import pickle
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
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from .data_loader import *

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512# suitable for all datasets
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
num_labels = 3

class MainModel(BertPreTrainedModel):
    "Main Model class"
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config (transformers.PretrainedConfig): Configuration for the model.
            loss_fn (custom, optional): Loss function to use. Default is None.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = 2
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased",config = config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor): Input IDs of a batch.
            attention_mask (torch.Tensor): Attention mask of a batch.
            token_type_ids (torch.Tensor): Token type IDs.
            labels (torch.Tensor): Target labels of a batch.
            device (str): Device to run the model on(gpu or cpu).
        
        Returns:
            output (torch.Tensor): Embeddings corresponding the [CLS] token of the finla layer.
        """
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
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
            

def read_dataset(data_path):
    """
    Reads a dataset object from a pickle file.
    
    Args:
        data_path (str): Path to the pickle file containing the dataset.
        
    Returns:
        data: The dataset loaded from the pickle file.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

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
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path} : {total_embeddings}')
    return embeddings

def main():
    """
    Main function to train and evaluate the model.
    """
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--train_file_path', type=str, required=True)
    parser.add_argument('--dev_file_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    model = MainModel.from_pretrained(args.input_model_path,num_labels = 2, loss_fn = None)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)


    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path

    data = read_dataset('../resources/QQP/train.pkl')
    dev_data = read_dataset('../resources/QQP/val.pkl')
    print(f'Train dataset len: {len(data)}')
    print(f'Validation dataet len: {len(dev_data)}')

    paws_data = load_paws(file_path='../resources/PAWS/test.tsv', tokenizer=tokenizer)
    paws_dataloader = DataLoader(paws_data, shuffle = False, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dev_data, shuffle=False, batch_size=BATCH_SIZE)
    
    qqp_train_embedding_path = './Embeddings/QQP/qqp_train_embeddings.txt'
    generate_embeddings(model, train_dataloader, device, qqp_train_embedding_path)
    

    paws_eval_embedding_path = './Embeddings/QQP/QQP_eval_embeddings.txt'
    generate_embeddings(model, eval_dataloader, device, paws_eval_embedding_path)

    paws_test_embedding_path = './Embeddings/PAWS/PAWS_test_embeddings.txt'
    generate_embeddings(model, paws_dataloader, device, paws_test_embedding_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()