"""
This script is designed to generate embeddings
from the model, and save these embeddings to text files.

Usage::

    python generate_embeddings.py --input_model_path <path_to_pretrained_model> --mnli_path <path_to_mnli_train_file> --mnli_val_path <path_to_mnli_val> --hans_test_path <hans_dataset_path>

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    --mnli_train_path (str): Path to the MNLI train dataset file.
    --mnli_val_path (str): Path to the MNLI validation dataset file.
    --hans_test_path (str): Path to the HANS test file.

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
from torch.utils.data import ConcatDataset
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from datasets import load_dataset


from data_loader import load_mnli, load_hans, load_snli

# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_mnli
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_hans
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_snli
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
    def __init__(self, config):
        """
        Initializes the MainModel.

        Args:
            config (BertConfig): The configuration object used to initialize the model.

        Attributes:
            num_labels (int): The number of labels for classification.
            model_l (BertModel): The left BERT model.
            model_b (BertModel): The right BERT model.
            fc_l_1 (Linear): The first fully connected layer for the left BERT model.
            fc_l_2 (Linear): The second fully connected layer for the left BERT model.
            fc_b_1 (Linear): The first fully connected layer for the right BERT model.
            fc_b_2 (Linear): The second fully connected layer for the right BERT model.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_l_2 = nn.Linear(2*(self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_b_2 = nn.Linear(2*(self.num_labels), self.num_labels)

    def features(self,input_ids, attention_mask, token_type_ids):
        """
        Extracts features from input using two BERT models.

        Args:
            input_ids (Tensor): The input token IDs.
            attention_mask (Tensor): The attention mask.
            token_type_ids (Tensor): The token type IDs.

        Returns:
            Tensor: Concatenated features from both BERT models.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_b = self.model_b(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_l = z_l.last_hidden_state[:,0,:]
        z_b = z_b.last_hidden_state[:,0,:]
        z = torch.cat((z_l, z_b), dim=1)
        return z

def save_embeddings_to_text_file(embeddings, output_file_path):
    """
    Saves embeddings to a text file.

    Args:
        embeddings (List[Tensor]): List of embedding tensors.
        output_file_path (str): The path to the output text file.
    """
    with open(output_file_path, 'a') as file:
        for embedding in embeddings:
            for i, emb in enumerate(embedding):
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
    Generates and saves embeddings from the model.

    Args:
        model (MainModel): The model to generate embeddings from.
        dataloader (DataLoader): The DataLoader object containing the dataset.
        device (str): The device to run the model on ('cuda' or 'cpu').
        output_file_path (str): The path to save the output embeddings text file.

    Returns:
        List[List[float]]: The list of embeddings generated.
    """
    embeddings = []
    total_embeddings = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        with torch.no_grad():
            output = model.features(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        embeddings = output.tolist()
        save_embeddings_to_text_file(embeddings, output_file_path)
        total_embeddings += len(embeddings)
    print(f'Total embeddings saved in {output_file_path} : {total_embeddings}')
    return embeddings


def main():
    """
    Main function to generate embeddings from the model and save them to files.

    This function loads the required datasets, initializes the model and tokenizer,
    generates embeddings for each dataset, and saves them to corresponding output files.

    Args:
        None

    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--mnli_matched_path', type=str, required=True)
    parser.add_argument('--mnli_mismatched_path', type=str, required=True)
    parser.add_argument('--hans_file1_path', type=str, required=True)
    parser.add_argument('--hans_file2_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    config = AutoConfig.from_pretrained(args.input_model_path , num_labels=num_labels)
    model = MainModel.from_pretrained(args.input_model_path, config = config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    
    #loding HANS data
    data1 = load_hans(file_path=args.hans_file1_path, tokenizer=tokenizer)
    data2 = load_hans(file_path=args.hans_file2_path, tokenizer=tokenizer)
    print(data1.len)
    print(data2.len)

    # data = ConcatDataset([data1, data2])
    hans_data = data1


    mnli_train_path = './Predictions/MNLI/mnli_train_embeddings.txt'
    train_data = load_mnli(file_path='../resources/multinli_1.0/multinli_1.0_train.txt', tokenizer=tokenizer)
    hans_test_dataloader = DataLoader(hans_data, shuffle = False, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
    generate_embeddings(model, train_dataloader, device, mnli_train_path)

    data = read_dataset('../resources/multinli_1.0/val.pkl')
    val_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    generate_embeddings(model, val_dataloader, device, './Predictions/MNLI/mnli_eval_embeddings.txt')

    mnli_test_data = read_dataset('../resources/multinli_1.0/test.pkl')
    mnli_test_dataloader = DataLoader(mnli_test_data, shuffle = False, batch_size=BATCH_SIZE)

    
    
    mnli_test_embedding_path = './Predictions/MNLI/MNLI_test_embeddings.txt'
    generate_embeddings(model, mnli_test_dataloader, device, mnli_test_embedding_path)

    hans_embedding_path = './Predictions/HANS/HANS_test_embeddings.txt'
    generate_embeddings(model, hans_test_dataloader, device, hans_embedding_path)


    snli_data = load_snli(file_path='../resources/snli_1.0/snli_1.0_test.txt', tokenizer=tokenizer)
    snli_test_dataloader = DataLoader(snli_data, shuffle = False, batch_size=BATCH_SIZE)
    snli_embedding_path = './Predictions/SNLI/SNLI_test_embeddings.txt'
    generate_embeddings(model, snli_test_dataloader, device, snli_embedding_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()

