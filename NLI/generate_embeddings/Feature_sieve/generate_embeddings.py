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
from .data_loader import load_mnli
from .data_loader import load_hans
from .data_loader import load_snli
from typing import List

# Ignore all warnings
warnings.filterwarnings("ignore")

MAX_LEN = 512  # Suitable for all datasets
LEARNING_RATE = 1e-5
NUM_LABELS = 3  # Define the number of labels
BATCH_SIZE = 32  # Define the batch size

# embedding_generator.py

class BiasModel(nn.Module):
    """Implements a bias model with attention mechanism for classification tasks.

    This model incorporates attention to compute a weighted sum of the input embeddings
    before making predictions.

    Args:
        attention_size (int): The size of the attention layer.
        hidden_size (int): The size of the hidden layer.
        num_labels (int): The number of output labels.

    Attributes:
        dropout (nn.Dropout): Dropout layer to regularize the input embeddings.
        attention1 (nn.Linear): Linear layer to transform input embeddings to attention space.
        attention2 (nn.Linear): Linear layer to compute attention scores.
        classifier (nn.Linear): Linear layer to produce logits for classification.

    """

    def __init__(self, attention_size, hidden_size, num_labels):
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size, attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
    def forward(self, bert_hidden_layer_input):
        """Defines the forward pass of the model.

        Args:
            bert_hidden_layer_input (torch.Tensor): The input hidden layer from a pre-trained BERT model.

        Returns:
            torch.Tensor: Logits for classification.

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
    """Implements a main model for a natural language inference task.

    This model integrates a BERT model with an additional bias model for classification.

    Args:
        config (BertConfig): The configuration object specifying the model architecture.
        loss_fn (optional): The loss function used for training. Defaults to None.

    Attributes:
        num_labels (int): The number of output labels.
        loss_fn (optional): The loss function used for training.
        bert (AutoModel): The pre-trained BERT model.
        bias_model (BiasModel): The bias model for incorporating additional information.
        hidden_layer (nn.Linear): The linear layer for the hidden states.
        classifier (nn.Linear): The linear layer for classification.

    """

    def __init__(self, config, loss_fn=None):
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)
        self.bias_model = BiasModel(attention_size=1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2 * (self.num_labels))
        self.classifier = nn.Linear(2 * (self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.

        Returns:
            torch.Tensor: The output representation of the input sequences.

        """
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output.hidden_states[4]
        hidden_state_detached = hidden_state.detach()
        bias_prob = self.bias_model(hidden_state_detached)
        bias_loss = F.cross_entropy(bias_prob, torch.ones_like(bias_prob).long())
        forget_prob = self.bias_model(hidden_state)
        pseudo_labels = torch.ones_like(forget_prob) / self.num_labels
        forget_loss = F.cross_entropy(forget_prob, pseudo_labels)
        output = output.last_hidden_state
        output = output[:, 0, :]
        return output




def read_dataset(data_path: str):
    """Reads a dataset from a pickle file.

    Args:
        data_path (str): The path to the pickle file containing the dataset.

    Returns:
        Any: The loaded dataset.

    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

def save_embeddings_to_text_file(embeddings: List[List[float]], output_file_path: str):
    """Saves embeddings to a text file.

    Args:
        embeddings (List[List[float]]): The embeddings to be saved.
        output_file_path (str): The path to the output text file.

    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            file.write(' '.join(map(str, embedding)) + '\n')

def generate_embeddings(model, dataloader, device, output_file_path):
    """Generates embeddings using the provided model and saves them to a text file.

    Args:
        model: The model used for generating embeddings.
        dataloader (DataLoader): The DataLoader containing the input data.
        device (torch.device): The device on which the model will be evaluated.
        output_file_path (str): The path to the output text file for saving embeddings.

    Returns:
        int: The total number of embeddings generated.

    """
    model.eval()
    total_embeddings = 0
    for batch in tqdm(dataloader, ncols=100):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        with torch.no_grad():
            embeddings = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        embeddings = embeddings.cpu().tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)

    print(f'Total embeddings saved in {output_file_path}: {total_embeddings}')
    return total_embeddings

def custom_collate_fn(batch):
    """Custom collate function for DataLoader.

    Args:
        batch (List[Dict]): A list of dictionaries containing batched data.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing batched tensors for 'ids', 'mask', 'token_type_ids', and 'target'.

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
    """Main function to generate embeddings for MNLI and SNLI datasets using a pre-trained model.

    This function loads a pre-trained model, generates embeddings for MNLI and SNLI datasets,
    and saves the embeddings to text files.

    """
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

