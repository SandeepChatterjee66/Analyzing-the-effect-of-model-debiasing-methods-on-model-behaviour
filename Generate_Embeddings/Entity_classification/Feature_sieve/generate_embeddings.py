
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
# from data_loader import load_data
# from data_loader import load_mapping
from Generate_Embeddings.Entity_classification.Feature_sieve.data_loader import load_data
from Generate_Embeddings.Entity_classification.Feature_sieve.data_loader import load_mapping

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
log_soft = F.log_softmax
print(torch.version.cuda)
MAX_LEN = 512# suitable for all datasets
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
num_labels = 0


class BiasModel(nn.Module):
    """
    Biased attention model for generating attention-based embeddings.

    Attributes:
        num_labels (int): Number of labels.
    """
    def __init__(self, attention_size, hidden_size, num_labels):
        """
        Initializes the BiasModel.

        Args:
            attention_size (int): Size of the attention.
            hidden_size (int): Size of the hidden layer.
            num_labels (int): Number of labels.
        """
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)  
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size,attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
    def forward(self, bert_hidden_layer_input):
        """
        Forward pass through the model.

        Args:
            bert_hidden_layer_input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits.
        """
        bert_hidden_layer_input = self.dropout(bert_hidden_layer_input)
        # print(bert_hidden_layer_input.shape)
        attention1_out = self.attention1(bert_hidden_layer_input)
        # print(attention1_out.shape)
        attention1_out = torch.tanh(attention1_out)
        # print(attention1_out.shape)
        attention2_out = self.attention2(attention1_out)
        # print(attention2_out.shape)
        # print(attention2_out)
        attention2_out = F.softmax(attention2_out, dim = 1)
        # print(attention2_out.shape)
        # print(attention2_out)
        # print(torch.sum(attention2_out, dim = 1))
        # print(attention2_out.unsqueeze(-1).shape)
        weighted_sum = torch.sum(attention2_out * bert_hidden_layer_input, dim=1)
        # print(weighted_sum.shape)
        logits = self.classifier(weighted_sum)
        # print(logits)
        return logits
        



class MainModel(BertPreTrainedModel):
    """
    Main model for generating embeddings using BERT.

    Attributes:
        num_labels (int): Number of labels.
    """
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config (AutoConfig): Configuration object.
            loss_fn (function): Loss function.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1",config = config)
        self.bias_model = BiasModel(attention_size = 1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, labels,device):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Labels.
            device (str): Device to run the model on.

        Returns:
            torch.Tensor: Output tensor.
        """
              
        output = self.bert(input_ids, attention_mask = attention_mask)
        hidden_state = output.hidden_states[4]
        hidden_state_detached = hidden_state.detach()
        bias_prob = self.bias_model(hidden_state_detached)
        bias_loss = F.cross_entropy(bias_prob, labels.view(-1))
        forget_prob = self.bias_model(hidden_state)
        pseudo_labels = torch.ones_like(forget_prob) / self.num_labels
        forget_loss =  F.cross_entropy(forget_prob, pseudo_labels)
        # print(hidden_state.shape)
        # output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:]
        
        return output

def save_embeddings_to_text_file(embeddings, output_file_path):
    """
    Saves embeddings to a text file.

    Args:
        embeddings (list): List of embeddings.
        output_file_path (str): Output file path.

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
    Generates embeddings using the provided model and saves them to a text file.

    Args:
        model (MainModel): Main model for generating embeddings.
        dataloader (DataLoader): DataLoader for loading data.
        device (str): Device to run the model on.
        output_file_path (str): Output file path.

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
    Main function to run the embedding generation process.

    This function performs the following steps:
    1. Parses command line arguments to get dataset names, model and tokenizer directories, mapping file, output file, and train file flag.
    2. Initializes the main model, tokenizer, and device.
    3. Loads the data and creates DataLoaders.
    4. Generates embeddings and saves them to a text file.
    5. Logs total training time.

    Args:
        --train_dataset_name (str): Name of the training dataset.
        --test_dataset_name (str): Name of the test dataset.
        --model_directory (str): Directory containing the model.
        --tokenizer_directory (str): Directory containing the tokenizer.
        --mapping_file (str): File path for the mapping file (optional).
        --output_file (str): Output file path.
        --train_file (str): Flag indicating whether to use the train file (optional).

    Returns:
        None
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
        id2label_test,_, test_data = load_data(args.test_dataset_name, 'test', tokenizer, cross_eval=True, mapping = mapping)
        test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
    else:
        if(args.train_file == 'True'):
            id2label, label2id, train_data = load_data(args.train_dataset_name, 'train', tokenizer)
            _,_, devel_data = load_data(args.train_dataset_name, 'devel', tokenizer)
            train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
            devel_dataloader = DataLoader(devel_data, shuffle = False, batch_size=BATCH_SIZE)
        else:
            id2label, label2id, test_data = load_data(args.test_dataset_name, 'test', tokenizer)
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