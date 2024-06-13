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

class mnli_dataset(Dataset):
    """
    Dataset class for the Multi-Genre Natural Language Inference (MNLI) dataset.
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
        label_dict = {'contradiction' : 0, 'neutral' : 1, 'entailment' : 2}
        label = label_dict[label]
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

def load_mnli(file_path,tokenizer, type = True):
    """
    Loads the MNLI dataset from a file and return the dataset object.

    Args:
        file_path (str): Path to the MNLI dataset file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        type (bool): Boolean indicating the type of MNLI data(train or dev) (type = True for train).

    Returns:
        data: The loaded MNLI dataset object.
    """
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            label = parts[0]
            if(type):
                sentence1 = parts[-6]
                sentence2 = parts[-5]
            else:
                sentence1 = parts[-10]
                sentence2 = parts[-9]
            if label == 'contradiction' or label == 'entailment' or label == 'neutral':
                target_label_list.append(label)
                sentence1_list.append(sentence1)
                sentence2_list.append(sentence2)

    print(len(sentence1_list))
    path = file_path.split('.')
    np.savetxt('./multinli_1.0/' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
    data = mnli_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data

class hans_dataset(Dataset):
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
        label_dict = {'non-entailment' : 1, 'entailment' : 2}
        label = label_dict[label]
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

def load_hans(file_path,tokenizer):
    """
    Loads the HANS dataset from a file.

    Args:
        file_path (str): Path to the HANS dataset file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        data: The loaded HANS dataset object.
    """
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            label = parts[0]
            sentence1 = parts[-6]
            sentence2 = parts[-5]
            if label == 'non-entailment' or label == 'entailment':
                target_label_list.append(label)
                sentence1_list.append(sentence1)
                sentence2_list.append(sentence2)

    print(len(sentence1_list))
    path = file_path.split('.')
    np.savetxt('.' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
    data = hans_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data


def load_snli(file_path,tokenizer):
    """
    Loads the SNLI dataset from a file.

    Args:
        file_path (str): Path to the SNLI dataset file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        data: The loaded SNLI dataset object.
    """
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            label = parts[0]
            sentence1 = parts[-9]
            sentence2 = parts[-8]
            
            if len(parts) == 10:
                sentence1 = parts[-5]
                sentence2 = parts[-4]
                # print(sentence1)
                # print(sentence2)
                # print(parts)
                # break
            elif len(parts) == 12:
                sentence1 = parts[-7]
                sentence2 = parts[-6]
                
            elif len(parts) == 13:
                sentence1 = parts[-8]
                sentence2 = parts[-7]
                
            elif len(parts) == 14:
                sentence1 = parts[-9]
                sentence2 = parts[-8]

            if label == 'contradiction' or label == 'entailment' or label == 'neutral':
                target_label_list.append(label)
                sentence1_list.append(sentence1)
                sentence2_list.append(sentence2)

    print(len(sentence1_list))
    path = file_path.split('.')
    np.savetxt('./snli_2.0/' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
    data = mnli_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data