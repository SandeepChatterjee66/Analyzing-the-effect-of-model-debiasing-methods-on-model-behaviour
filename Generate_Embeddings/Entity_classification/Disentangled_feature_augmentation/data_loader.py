"""
This script contains classes and functions for loading and processing datasets
for entity classification tasks.
"""
from multiprocessing import reduction
import pandas as pd
import time
import numpy as np
import csv
import argparse
import math
from sklearn.metrics import accuracy_score
import torch
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

input_path = './'
MAX_LEN = 512   # suitable for all datasets

def read_data(dataset_name, data):
    """
    Read and parse the dataset to extract sentences and their labels.

    Args:
    
        dataset_name (str): Name of the dataset(can be MedMentions, BC5CDR, NCBI_disease).
        data (str): Name of the data file (can be train, devel or test)

    Returns:
        tuple: A tuple containing lists of sentences, labels, and all_labels.
    """
    data = data + '.txt'
    path = os.path.join('../resources', dataset_name, data)
    token_lst, label_lst = [], []
    sentences = []
    labels = []
    all_labels = []
    with open(path, 'r') as fh:
        for line in fh:
            if len(line.strip()) == 0:
                sentences.append(token_lst)
                labels.append(label_lst)
                token_lst = []
                label_lst = []
                continue
            a = line.split('\t')
            token_lst.append(a[0].strip())
            label_lst.append(a[1].strip())
    label_path = 'labels.txt'
    label_path = os.path.join(input_path, dataset_name, label_path)
    with open(label_path, 'r') as fh:
        for line in fh:
            if(len(line.strip()) != 0):
                all_labels.append(line.strip())
    if(dataset_name == 'MedMentions'):
        assert(len(all_labels) == 128)
    return sentences, labels, all_labels


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
    """
    Tokenize sentences while preserving the original labels for each token.

    Args:
        sentence (list): List of tokens in a sentence.
        text_labels (list): List of labels (B-<CLASS>, I, O) corresponding to each token in the sentence.
        tokenizer: Tokenizer object used for tokenization.

    Returns:
        tuple: A tuple containing the tokenized sentence and the corresponding labels after tokenization.
    """
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):
        tokenized_word = tokenizer.tokenize(word)
        if(len(tokenized_word) == 0 and len(word) != 0):
            tokenized_word = ' '
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)
        # add same label of word to other subwords
        if label[0] != 'O':
            labels.extend([label])
            new_label = 'I' + label[1:]
            labels.extend([new_label] * (n_subwords-1))
        else:
            labels.extend(label * (n_subwords))

    return tokenized_sentence, labels 


def IdToLabelAndLabeltoId(label_list):
    """
    Create mappings from label to ID and ID to label for the dataset.

    Args:
        label_list (list): List of all unique labels in the dataset.

    Returns:
        tuple: A tuple containing two dictionaries, id2label and label2id.
    """
    label_list = [x for x in label_list if not pd.isna(x)]
    # sorting as applying set operation does not maintain the order
    label_list.sort()
    id2label = {}
    for index, label in enumerate(label_list):
        id2label[index] = label
    label2id = { id2label[id]: id for id in id2label}
    return id2label,label2id


class dataset(Dataset):
    def __init__(self, sentence_list, labels_list, tokenizer, max_len, label2id, id2label):
        """
        Custom dataset class.

        Args:
            sentence_list (list): List of tokenized sentences.
            labels_list (list): List of labels corresponding to each token in the sentence.
            tokenizer: Tokenizer object used for tokenization.
            max_len (int): Maximum length of tokens.
            label2id (dict): Dictionary mapping labels to IDs.
            id2label (dict): Dictionary mapping IDs to labels.
        """
        self.len = len(sentence_list)
        self.sentence = sentence_list
        self.labels = labels_list
        self.tokenizer = tokenizer
        self.max_len = max_len 
        self.label2id = label2id
        self.id2label = id2label
        self.maximum_across_all = 0 

    def __getitem__(self, index):
        """
        Get a single data point from the dataset.

        Args:
            index (int): Index of the data point.

        Returns:
            dict: A dictionary containing index, token IDs, attention mask, and target labels.
        """
        # step 1: tokenize sentence and adapt labels
        sentence = self.sentence[index]
        label = self.labels[index]
        label2id = self.label2id
        tokenized_sentence = ['[CLS]'] + sentence + ['[SEP]']
        
        # step 3: truncating or padding
        max_len = self.max_len
        #print(tokenized_sentence)
        if len(tokenized_sentence) > max_len:
            #truncate
            tokenized_sentence = tokenized_sentence[:max_len]
        else:
            # pad
            tokenized_sentence = tokenized_sentence + ['[PAD]' for _ in range(max_len - len(tokenized_sentence))]

        # step 4: obtain attention mask
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in tokenized_sentence]

        # step 5: convert tokens to input ids
        ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_id = label2id[label]
        target = []
        target.append(label_id)

        return {
            'index': index,
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

    def __len__(self):
        return self.len

def load_mapping(mapping_file):
    """
    Load mapping(needed when the model is trained on MedMentions and the test dataset is BC5CDR or NCBI disease) from a file.

    Args:
        mapping_file (str): Path to the mapping file.

    Returns:
        dict: A dictionary containing the mapping(MedMentions to BC5CDR or MedMentions to NCBI-disease).
    """
    mapping = {}
    with open(mapping_file, 'r') as fh:
        for line in fh:
            line = line.split('\t')
            mapping[line[0].strip('\n')] = line[1].strip('\n')
    return mapping


def load_data(dataset_name, data, tokenizer, cross_eval = False, mapping = None):
    """
    Load and preprocess data for training.

    Args:
        dataset_name (str): Name of the dataset.
        data (str): Name of the data file (can be train, devel or test).
        tokenizer: Tokenizer object used for tokenization.
        cross_eval (bool): Flag indicating if cross evaluation is to be performed(needed when train and test dataset are different. default = False).
        mapping (dict): Optional mapping dictionary(needed if classes are different in the train and test dataset).

    Returns:
        tuple: A tuple containing id2label, label2id, dataset object, and labels list.
    """
    sentences,labels, all_labels = read_data(dataset_name, data)
    sentence_list = []
    count = 0
    labels_list = []
    for sentence,label in zip(sentences,labels):
        entity_list = []
        tokenized_sentence,tokenized_label = tokenize_and_preserve_labels(sentence, label, tokenizer)
        
        if len(tokenized_sentence) > 512:
            count += 1
        for i in range(len(tokenized_label)):
            if(tokenized_label[i][0] == 'B'):
                l = i
                entity_class = tokenized_label[i][2:]
                i += 1
                while(i < len(tokenized_label) and tokenized_label[i][0] == 'I'):
                    i += 1
                r = i
                i -= 1
                if cross_eval == True and dataset_name == 'MedMentions':
                    if mapping.get(entity_class) is not None:
                        entity_list.append([l,r,entity_class])
                else:
                    entity_list.append([l,r,entity_class])
        for entity in entity_list:
            new_sent = []
            if(len(tokenized_sentence) > 510):
                entity_len = entity[1] - entity[0]
                prefix_len = (510 - entity_len)//2
                suffix_len = prefix_len
                if(prefix_len < entity[0] and suffix_len < (len(tokenized_sentence) - entity[1])):
                    start = entity[0] - prefix_len
                    end = entity[1] + suffix_len
                elif(prefix_len >= entity[0]):
                    start = 0
                    suffix_len += (prefix_len - entity[0])
                    end = entity[1] + suffix_len
                elif(suffix_len >= len(tokenized_sentence) - entity[1] ):
                    end = len(tokenized_sentence)
                    prefix_len += (suffix_len - len(tokenized_sentence) + entity[1])
                    start = entity[0] - prefix_len
            else:
                start = 0
                end = len(tokenized_sentence)

            for i in range(start,end):
                if(i == entity[0]):
                    new_sent.append('[ENTITY_START]')
                elif(i == entity[1]):
                    new_sent.append('[ENTITY_END]')
                new_sent.append(tokenized_sentence[i])
            if(entity[1] == len(tokenized_label)):
                new_sent.append('[ENTITY_END]')
            assert(len(new_sent) <= 512)
            sentence_list.append(new_sent)
            labels_list.append(entity[2])
    print(len(sentence_list))
    print(len(labels_list))
  
    id2label,label2id = IdToLabelAndLabeltoId(all_labels)
    data  = dataset(sentence_list=sentence_list, labels_list=labels_list, tokenizer=tokenizer,\
                          max_len=MAX_LEN, id2label=id2label, label2id=label2id)
    
    print(f'Sentence with more than 512 tokens : {count} \nTotal sentences : {len(sentences)}')
    return id2label, label2id, data, labels_list
