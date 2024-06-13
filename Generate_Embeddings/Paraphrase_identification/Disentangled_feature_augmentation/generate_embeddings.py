"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> 

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    
"""

from multiprocessing import reduction
import pandas as pd
import time
import numpy as np
import csv
import argparse
import math
import gc
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

from data_loader import load_qqp, load_paws
# from Generate_Embeddings.Paraphrase_identification.Disentangled_feature_augmentation.data_loader import load_qqp, load_paws

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
    Main model for pre-training BERT for downstream tasks.
    
    Args:
        config (obj): Configuration for the BERT model.
    """
    def __init__(self, config):
        super(MainModel,self).__init__(config)
        self.num_labels = 2#config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_l_2 = nn.Linear(2*(self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_b_2 = nn.Linear(2*(self.num_labels), self.num_labels)

    def features(self,input_ids, attention_mask):
        """
        Get features from the BERT models.
        
        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            
        Returns:
            tuple: Tuple containing the features from the two BERT models.
        """
        z_l = self.model_l(input_ids, attention_mask = attention_mask)
        z_b = self.model_b(input_ids, attention_mask = attention_mask)
        return z_l.last_hidden_state[:,0,:], z_b.last_hidden_state[:,0,:]
    
    def linear(self, z_conflict, z_align):
        """
        Linear layers for conflict and alignment features.
        
        Args:
            z_conflict (torch.Tensor): Features from the conflict BERT model.
            z_align (torch.Tensor): Features from the alignment BERT model.
            
        Returns:
            tuple: Tuple containing the outputs from the linear layers.
        """
        hidden_output1 = self.fc_l_1(z_conflict)
        output1 = self.fc_l_2(hidden_output1)
        hidden_output2 = self.fc_b_1(z_align)
        output2 = self.fc_b_2(hidden_output2)
        return output1,output2
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Input token ids.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type ids.
            
        Returns:
            torch.Tensor: Output of the forward pass.
        """
        z_conflict, z_align = self.encode(input_ids, attention_mask, token_type_ids)  # Assuming these are the functions
        z_combined = torch.cat((z_conflict, z_align), dim=1)
        hidden_output1 = self.fc_l_1(z_combined)
        return hidden_output1

def read_dataset(data_path: str):
    """
    Read dataset from a pickle file.
    
    Args:
        data_path (str): Path to the pickle file containing the dataset.
        
    Returns:
        obj: Loaded dataset.
    """
    import pickle
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
def save_embeddings_to_text_file(embeddings, output_file_path):
    """
    Save embeddings to a text file.
    
    Args:
        embeddings (list): List of embeddings.
        output_file_path (str): Path to the output text file.
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
    Generate embeddings and save them to a text file.
    
    Args:
        model (obj): Model for generating embeddings.
        dataloader (obj): DataLoader for the dataset.
        device (str): Device to run the model on.
        output_file_path (str): Path to the output text file.
        
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
            output = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path} : {total_embeddings}')
    return embeddings

def main():

    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path , num_labels=num_labels)
    model = MainModel.from_pretrained(args.input_model_path, config = config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    train_data = read_dataset('../resources/QQP/train.pkl')
    dev_data = read_dataset('../resources/QQP/val.pkl')


    print(len(train_data), "train data length")
    train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
    print("Testing started for qqp train")
    qqp_train_embedding_path = './Predictions/QQP/qqp_train_embeddings.txt'
    generate_embeddings(model, train_dataloader, device, qqp_train_embedding_path)


    print(len(dev_data), "dev data length")
    dev_dataloader = DataLoader(dev_data, shuffle = False, batch_size=BATCH_SIZE)
    print("Testing started for qqp train")
    qqp_dev_embedding_path = './Predictions/QQP/qqp_dev_embeddings.txt'
    generate_embeddings(model, dev_dataloader, device, qqp_dev_embedding_path)

    #loding PAWS data
    data_paws_dev = load_paws(file_path='../resources/PAWS/dev.tsv', tokenizer=tokenizer)
    print(data_paws_dev)
    data_paws_test = load_paws(file_path='../resources/PAWS/test.tsv', tokenizer=tokenizer)
    print(data_paws_test)
    data_paws_train = load_paws(file_path='../resources/PAWS/train.tsv', tokenizer=tokenizer)
    print(data_paws_train)

    paws_dev_dataloader = DataLoader(data_paws_dev, shuffle = False, batch_size=BATCH_SIZE)
    paws_test_dataloader = DataLoader(data_paws_test, shuffle = False, batch_size=BATCH_SIZE)
    paws_train_dataloader = DataLoader(data_paws_train, shuffle = False, batch_size=BATCH_SIZE)
    
    paws_dev_embedding_path = './Predictions/PAWS/paws_dev_embeddings.txt'
    generate_embeddings(model, paws_dev_dataloader, device, paws_dev_embedding_path)
    paws_test_embedding_path = './Predictions/PAWS/paws_test_embeddings.txt'
    generate_embeddings(model, paws_test_dataloader, device, paws_test_embedding_path)
    paws_train_embedding_path = './Predictions/PAWS/paws_train_embeddings.txt'
    generate_embeddings(model, paws_train_dataloader, device, paws_train_embedding_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()