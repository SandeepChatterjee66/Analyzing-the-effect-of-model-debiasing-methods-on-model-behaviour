"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> 

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    
"""
import os
import time
import pickle
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel, BertPreTrainedModel
from tqdm import tqdm
from Generate_Embeddings.Paraphrase_identification.LfF.data_loader import load_paws
from typing import List

warnings.filterwarnings("ignore")

MAX_LEN = 512  # Suitable for all datasets
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_LABELS = 3

class MainModel(BertPreTrainedModel):
    """Main Model class inheriting from BertPreTrainedModel."""

    def __init__(self, config):
        """
        Initializes the MainModel.

        Args:
            config (PretrainedConfig): Configuration for the model.
        """
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2 * self.num_labels)
        self.fc_l_2 = nn.Linear(2 * self.num_labels, self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2 * self.num_labels)
        self.fc_b_2 = nn.Linear(2 * self.num_labels, self.num_labels)

    def features(self, input_ids, attention_mask, token_type_ids):
        """
        Extracts features from the input.

        Args:
            input_ids (Tensor): Input IDs.
            attention_mask (Tensor): Attention mask.
            token_type_ids (Tensor): Token type IDs.

        Returns:
            Tensor: Concatenated features from model_l and model_b.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_b = self.model_b(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_l = z_l.last_hidden_state[:, 0, :]
        z_b = z_b.last_hidden_state[:, 0, :]
        z = torch.cat((z_l, z_b), dim=1)
        return z

def read_dataset(data_path: str):
    """
    Reads dataset from file.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        Any: Loaded dataset.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

def save_embeddings_to_text_file(embeddings: List[List[float]], output_file_path: str):
    """
    Saves embeddings to a text file.

    Args:
        embeddings (List[List[float]]): List of embeddings.
        output_file_path (str): Path to the output text file.
    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            file.write(' '.join(map(str, embedding)) + '\n')

def generate_embeddings(model, dataloader, device, output_file_path):
    """
    Generates embeddings using the provided model and saves them to a text file.

    Args:
        model (MainModel): Model to generate embeddings.
        dataloader (DataLoader): DataLoader for the dataset.
        device (str): Device to run the inference on (e.g., 'cpu', 'cuda').
        output_file_path (str): Path to save the embeddings text file.
    
    Returns:
        int: Total number of embeddings generated.
    """
    total_embeddings = 0
    for batch in tqdm(dataloader, ncols=100):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        with torch.no_grad():
            output = model.features(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        embeddings = output.cpu().tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)

    print(f'Total embeddings saved in {output_file_path}: {total_embeddings}')
    return total_embeddings

def custom_collate_fn(batch):
    """
    Custom collate function for the DataLoader.

    Args:
        batch (List): List of data samples.

    Returns:
        Dict: Dictionary containing batched data.
    """
    ids = torch.stack([item['ids'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])

    return {
        'ids': ids,
        'mask': masks,
        'token_type_ids': token_type_ids,
        'target': targets
    }

def main():
    """
    Main function for training the model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path, num_labels=NUM_LABELS)
    model = MainModel.from_pretrained(args.input_model_path, config=config)
    model.to(device)

    train_data = read_dataset('../resources/QQP/train.pkl')
    dev_data = read_dataset('../resources/QQP/val.pkl')

    print(len(train_data), "train data length")
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE)
    print("Testing started for qqp train")
    qqp_train_embedding_path = './Predictions/QQP/qqp_train_embeddings.txt'
    generate_embeddings(model, train_dataloader, device, qqp_train_embedding_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path, num_labels=NUM_LABELS)
    model = MainModel.from_pretrained(args.input_model_path, config=config)
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
    start = time.time()
    main()
