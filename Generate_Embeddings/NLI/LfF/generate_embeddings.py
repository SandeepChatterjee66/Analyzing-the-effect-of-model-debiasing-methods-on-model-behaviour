"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> --mnli_path <path_to_mnli_train_file> --mnli_val_path <path_to_mnli_val> --hans_test_path <hans_dataset_path>

Arguments::

    Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    --mnli_train_path (str): Path to the MNLI train dataset file.
    --mnli_val_path (str): Path to the MNLI validation dataset file.
    --hans_test_path (str): Path to the HANS test file.

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
from Generate_Embeddings.NLI.LfF.data_loader import load_mnli
from Generate_Embeddings.NLI.LfF.data_loader import load_hans
from Generate_Embeddings.NLI.LfF.data_loader import load_snli
from typing import List

# Ignore all warnings
warnings.filterwarnings("ignore")

MAX_LEN = 512  # Suitable for all datasets
LEARNING_RATE = 1e-5
NUM_LABELS = 3  # Define the number of labels
BATCH_SIZE = 32  # Define the batch size

class MainModel(BertPreTrainedModel):
    """
    Main Model class for generating embeddings.
    """

    def __init__(self, config):
        """
        Initializes the MainModel.

        Args:
        config (BertConfig): The configuration object for the model.
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
        input_ids (torch.Tensor): Input token ids.
        attention_mask (torch.Tensor): Attention mask.
        token_type_ids (torch.Tensor): Token type ids.

        Returns:
        torch.Tensor: Concatenated features.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_b = self.model_b(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_l = z_l.last_hidden_state[:, 0, :]
        z_b = z_b.last_hidden_state[:, 0, :]
        z = torch.cat((z_l, z_b), dim=1)
        return z




def read_dataset(data_path: str):
    """
    Read a dataset from a pickle file.

    Args:
    data_path (str): The path to the pickle file containing the dataset.

    Returns:
    Any: The loaded dataset.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

def save_embeddings_to_text_file(embeddings: List[List[float]], output_file_path: str):
    """
    Save embeddings to a text file.

    Args:
    embeddings (List[List[float]]): List of embeddings.
    output_file_path (str): The path to save the embeddings text file.
    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            file.write(' '.join(map(str, embedding)) + '\n')

def generate_embeddings(model, dataloader, device, output_file_path):
    """
    Generate embeddings using the provided model and save them to a text file.

    Args:
    model: The model used for generating embeddings.
    dataloader: The dataloader providing data for generating embeddings.
    device: The device to run the model on.
    output_file_path (str): The path to save the embeddings text file.

    Returns:
    int: The total number of embeddings saved.
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
    Custom collate function for DataLoader to handle batches of variable-sized input sequences.

    Args:
    batch (List[Dict]): A list of dictionaries containing batched data.

    Returns:
    Dict: A dictionary containing batched tensors.
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path, num_labels=NUM_LABELS)
    model = MainModel.from_pretrained(args.input_model_path, config=config)
    model.to(device)

    # Generate embeddings for MNLI
    mnli_train_path = './Predictions/MNLI/mnli_train_embeddings.txt'
    train_data, _, _ = load_mnli(file_path='../resources/multinli_1.0/multinli_1.0_train.txt', tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    generate_embeddings(model, train_dataloader, device, mnli_train_path)

    # Generate embeddings for SNLI test
    snli_data = load_snli(file_path='../resources/snli_1.0/snli_1.0_test.txt', tokenizer=tokenizer)
    snli_test_dataloader = DataLoader(snli_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    snli_embedding_path = './Predictions/SNLI/SNLI_test_embeddings.txt'
    generate_embeddings(model, snli_test_dataloader, device, snli_embedding_path)
    

    # Generate embeddings for MNLI validation
    val_data = read_dataset('../resources/multinli_1.0/val.pkl')
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    generate_embeddings(model, val_dataloader, device, './Predictions/MNLI/mnli_eval_embeddings.txt')

    # Generate embeddings for MNLI test
    mnli_test_data = read_dataset('../resources/multinli_1.0/test.pkl')
    mnli_test_dataloader = DataLoader(mnli_test_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)
    mnli_test_embedding_path = './Predictions/MNLI/MNLI_test_embeddings.txt'
    generate_embeddings(model, mnli_test_dataloader, device, mnli_test_embedding_path)

    

    total_time = time.time() - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total running time: {total_time}\n')

    print(f"Total running time: {total_time}")

if __name__ == '__main__':
    start = time.time()
    main()

