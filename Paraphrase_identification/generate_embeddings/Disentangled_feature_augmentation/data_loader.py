"""
This script contains classes and functions for loading and processing datasets
for question pair classification tasks.
"""
import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import os
import gc
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
import numpy as np

MAX_LEN = 512

def tokenize_sent(sentence, tokenizer):
    """
    Tokenizes a given sentence using the provided tokenizer.

    Args:
        sentence (str): The sentence to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        list: A list of tokenized words.
    """
    tokenized_sentence = []
    sentence = str(sentence).strip()

    for word in sentence.split():
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence

class qqp_dataset(Dataset):
    """
    Dataset class for the Quora Question Pairs(QQP) dataset.
    """
    def __init__(self, sentence1, sentence2, label, tokenizer, max_len):
        """
        Initializes the dataset object with sentences, labels, tokenizer, and maximum length.

        Args:
            sentence1 (list): List of first sentences.
            sentence2 (list): List of second sentences.
            label (list): List of labels corresponding to the sentence pairs.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_len (int): Maximum length of tokenized sequences.
        """
        self.len = len(sentence1)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.max_len = max_len
        self.tokenizer = tokenizer
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing tokenized input IDs, attention mask, token type IDs, and target label.
        """
        sent1 = self.sentence1[idx]
        sent2 = self.sentence2[idx]
        label = self.label[idx]
        target = []
        target.append(label)
        
        token_type_ids = []
        token_type_ids.append(0)
        sent1 = tokenize_sent(sent1,self.tokenizer)
        sent2 = tokenize_sent(sent2,self.tokenizer)
        for i in enumerate(sent1):
            token_type_ids.append(0)
        token_type_ids.append(1)
        for i in enumerate(sent2):
            token_type_ids.append(1)
        token_type_ids.append(1)
        
        
        input_sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
        input_sent = input_sent + ['[PAD]' for _ in range(self.max_len - len(input_sent))]
        token_type_ids = token_type_ids + [0 for _ in range(self.max_len - len(token_type_ids))]
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in input_sent]
        ids = self.tokenizer.convert_tokens_to_ids(input_sent)
        return {
            'index' : idx,
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def load_qqp(file_path,tokenizer):
    """
    Loads the QQP dataset from a file and return the dataset object.

    Args:
        file_path (str): Path to the QQP dataset file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        
    Returns:
        data: The loaded QQP dataset object.
    """
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path)
    sentence1_list = dataframe['question1'].tolist()
    sentence2_list = dataframe['question2'].tolist()
    target_label_list = dataframe['is_duplicate'].tolist()

    path = file_path.split('.')
    np.savetxt('./QQP/qqp_groundtruth.txt', target_label_list, '%s')

    print(sentence1_list[0])
    print(len(sentence1_list))
    print(sentence2_list[0])
    print(len(sentence2_list))
    data = qqp_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data

class paws_dataset(Dataset):
    """
    Dataset class for the HANS dataset.
    """
    def __init__(self, sentence1, sentence2, label, tokenizer, max_len):
        """
        Initializes the dataset object with sentences, labels, tokenizer, and maximum length.

        Args:
            sentence1 (list): List of first sentences.
            sentence2 (list): List of second sentences.
            label (list): List of labels corresponding to the sentence pairs.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_len (int): Maximum length of tokenized sequences.
        """
        self.len = len(sentence1)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.max_len = max_len
        self.tokenizer = tokenizer
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.len

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing tokenized input IDs, attention mask, token type IDs, and target label.
        """
        sent1 = self.sentence1[idx]
        sent2 = self.sentence2[idx]
        label = self.label[idx]
        target = []
        target.append(label)
        
        token_type_ids = []
        token_type_ids.append(0)
        sent1 = tokenize_sent(sent1,self.tokenizer)
        sent2 = tokenize_sent(sent2,self.tokenizer)
        for i in enumerate(sent1):
            token_type_ids.append(0)
        token_type_ids.append(1)
        for i in enumerate(sent2):
            token_type_ids.append(1)
        token_type_ids.append(1)
        
        
        input_sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
        input_sent = input_sent + ['[PAD]' for _ in range(self.max_len - len(input_sent))]
        token_type_ids = token_type_ids + [0 for _ in range(self.max_len - len(token_type_ids))]
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in input_sent]
        ids = self.tokenizer.convert_tokens_to_ids(input_sent)
        return {
            'index' : idx,
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def load_paws(file_path,tokenizer):
    """
    Loads the PAWS dataset from a file.

    Args:
        file_path (str): Path to the PAWS dataset file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        data: The loaded PAWS dataset object.
    """
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path, delimiter='\t')
    sentence1_list = dataframe['sentence1'].tolist()
    sentence2_list = dataframe['sentence2'].tolist()
    target_label_list = dataframe['label'].tolist()

    path = file_path.split('.')
    #saving the ground truth
    np.savetxt('.' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
    print(sentence1_list[0])
    print(len(sentence1_list))
    print(sentence2_list[0])
    print(len(sentence2_list))
    data = paws_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data