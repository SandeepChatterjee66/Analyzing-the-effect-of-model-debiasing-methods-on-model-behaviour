"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> 

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    
"""
import pandas as pd
import time
import numpy as np
import csv
import gc
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
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from Generate_Embeddings.Paraphrase_identification.Feature_sieve.data_loader import load_paws
import pickle

# Ignore all warnings
warnings.filterwarnings("ignore")

input_path = './'
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512  # suitable for all datasets
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
num_labels = 3

class BiasModel(nn.Module):
    def __init__(self, attention_size, hidden_size, num_labels):
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size, attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, bert_hidden_layer_input):
        """
        Forward pass of the bias model.

        Args:
            bert_hidden_layer_input (torch.Tensor): Input tensor to the bias model.

        Returns:
            torch.Tensor: Output logits of the bias model.
        """
        bert_hidden_layer_input = self.dropout(bert_hidden_layer_input)
        attention1_out = self.attention1(bert_hidden_layer_input)
        attention1_out = torch.tanh(attention1_out)
        attention2_out = self.attention2(attention1_out)
        attention2_out = F.softmax(attention2_out, dim=1)
        weighted_sum = torch.sum(attention2_out * bert_hidden_layer_input, dim=1)
        logits = self.classifier(weighted_sum)
        return logits

class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn=None):
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)
        self.bias_model = BiasModel(attention_size=1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2 * (self.num_labels))
        self.classifier = nn.Linear(2 * (self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, device=None):
        """
        Forward pass of the main model.

        Args:
            input_ids (torch.Tensor): Input tensor of token indices.
            attention_mask (torch.Tensor): Input tensor of attention mask.
            token_type_ids (torch.Tensor): Input tensor of token type ids.
            labels (torch.Tensor, optional): Input tensor of labels for computing loss.
            device (str, optional): Device to run the computation on.

        Returns:
            torch.Tensor: Output tensor of the main model.
        """
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output.hidden_states[4]
        hidden_state_detached = hidden_state.detach()
        bias_prob = self.bias_model(hidden_state_detached)
        if labels is not None:
            bias_loss = F.cross_entropy(bias_prob, labels.view(-1))
            forget_prob = self.bias_model(hidden_state)
            pseudo_labels = torch.ones_like(forget_prob) / self.num_labels
            forget_loss = F.cross_entropy(forget_prob, pseudo_labels)
        output = output.last_hidden_state
        output = output[:, 0, :]
        return output

def read_dataset(data_path: str):
    """
    Reads dataset from the provided path.

    Args:
        data_path (str): Path to the dataset pickle file.

    Returns:
        list: The dataset loaded from the pickle file.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

def save_embeddings_to_text_file(embeddings, output_file_path):
    """
    Saves embeddings to a text file.

    Args:
        embeddings (list): List of embeddings to be saved.
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
    """
    Generates embeddings from the provided model and dataloader.

    Args:
        model (MainModel): The model to generate embeddings from.
        dataloader (DataLoader): The dataloader for loading the dataset.
        device (str): The device to run the computation on.
        output_file_path (str): Path to the output text file for saving embeddings.

    Returns:
        list: The embeddings generated.
    """
    total_embeddings = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=None, device=device)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path}: {total_embeddings}')
    return embeddings

def main():
    """
    Main function to orchestrate the generation of embeddings.

    Loads pretrained model, tokenizer, and dataset. Generates embeddings for train and dev datasets,
    and saves them to text files. Also generates embeddings for PAWS dataset.

    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path, num_labels=num_labels)
    model = MainModel.from_pretrained(args.input_model_path, config=config, loss_fn=None, ignore_mismatched_sizes=True)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)

    train_data = read_dataset('../resources/QQP/train.pkl')
    dev_data = read_dataset('../resources/QQP/val.pkl')

    print(len(train_data), "train data length")
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
    print("Testing started for qqp train")
    qqp_train_embedding_path = './Predictions/QQP/qqp_train_embeddings.txt'
    generate_embeddings(model, train_dataloader, device, qqp_train_embedding_path)

    print(len(dev_data), "dev data length")
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=BATCH_SIZE)
    print("Testing started for qqp train")
    qqp_dev_embedding_path = './Predictions/QQP/qqp_dev_embeddings.txt'
    generate_embeddings(model, dev_dataloader, device, qqp_dev_embedding_path)

    # Loading PAWS data
    data_paws_dev = load_paws(file_path='../resources/PAWS/dev.tsv', tokenizer=tokenizer)
    data_paws_test = load_paws(file_path='../resources/PAWS/test.tsv', tokenizer=tokenizer)
    data_paws_train = load_paws(file_path='../resources/PAWS/train.tsv', tokenizer=tokenizer)

    paws_dev_dataloader = DataLoader(data_paws_dev, shuffle=False, batch_size=BATCH_SIZE)
    paws_test_dataloader = DataLoader(data_paws_test, shuffle=False, batch_size=BATCH_SIZE)
    paws_train_dataloader = DataLoader(data_paws_train, shuffle=False, batch_size=BATCH_SIZE)

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
