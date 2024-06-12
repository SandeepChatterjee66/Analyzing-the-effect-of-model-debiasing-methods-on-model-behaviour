import os 
import time
import math 
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification,BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F
from seqeval.metrics import classification_report
import torch.nn as nn
from Entity_classification.generate_embeddings.Feature_sieve.train import MainModel
from Entity_classification.generate_embeddings.Feature_sieve.data_loader import *

BATCH_SIZE = 36
input_path = './'


def read_train_labels(dataset_name):
    """Read and return the training labels from a specified dataset.

    Args:
        dataset_name (str): The name of the dataset to read labels from.

    Returns:
        list: A list of labels read from the dataset.

    Raises:
        AssertionError: If the dataset is 'MedMentions' and the number of labels is not 128.
    """
    all_labels = []
    path = os.path.join(input_path, dataset_name, 'labels.txt')
    with open(path, 'r') as fh:
        for line in fh:
            if len(line.strip()) != 0:
                all_labels.append(line.strip())
    if dataset_name == 'MedMentions':
        assert len(all_labels) == 128
    return all_labels


def find_new_labels_MM_BC5(id2label, label2id, mapping, targets, predicted_labels):
    """Map predicted labels to new labels based on a provided mapping for the MM and BC5 datasets.

    Args:
        id2label (dict): Dictionary mapping label IDs to label names.
        label2id (dict): Dictionary mapping label names to label IDs.
        mapping (dict): Dictionary mapping original labels to new labels.
        targets (tensor): Tensor of true labels.
        predicted_labels (tensor): Tensor of predicted labels.

    Returns:
        tuple: A tuple containing:
            - list: A list of new labels.
            - float: The temporary test accuracy score.
    """
    new_labels = []
    labels = [id2label[class_id.item()] for class_id in predicted_labels]
    for label in labels:
        if mapping.get(label) is not None:
            new_labels.append(mapping[label])
        else:
            new_labels.append('nan')
    predicted_labels = [2 if label == 'nan' else label2id[label] for label in new_labels]
    tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), np.array(predicted_labels))
    return new_labels, tmp_test_accuracy


def find_new_target(id2label, label2id, mapping, targets):
    """Map true labels to new target labels based on a provided mapping.

    Args:
        id2label (dict): Dictionary mapping label IDs to label names.
        label2id (dict): Dictionary mapping label names to label IDs.
        mapping (dict): Dictionary mapping original labels to new labels.
        targets (tensor): Tensor of true labels.

    Returns:
        tensor: A tensor of new target labels, moved to the appropriate device.
    """
    targets = [id2label[target.item()] for target in targets]
    new_labels = [mapping[target] for target in targets]
    new_targets = torch.tensor([label2id[label] for label in new_labels])
    device = 'cuda' if cuda.is_available() else 'cpu'
    new_targets = new_targets.to(device, dtype=torch.long)
    return new_targets.unsqueeze(1)


def inference_MM_BC5(train_dataset, test_dataset, model, dataloader, tokenizer, device, id2label, label2id, mapping, id2label_train=None):
    """Perform inference on the MM and BC5 datasets using a trained model.

    Args:
        train_dataset (str): Name of the training dataset.
        test_dataset (str): Name of the test dataset.
        model (torch.nn.Module): The trained model to use for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for processing text.
        device (torch.device): Device to run the inference on.
        id2label (dict): Dictionary mapping label IDs to label names for the test dataset.
        label2id (dict): Dictionary mapping label names to label IDs for the test dataset.
        mapping (dict): Dictionary mapping original labels to new labels.
        id2label_train (dict, optional): Dictionary mapping label IDs to label names for the training dataset. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - float: The test accuracy.
            - list: A list of predicted labels.
            - list: A list of prediction probabilities.
    """
    model.eval()
    pred_lst = []
    prob_lst = []
    test_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        with torch.no_grad():
            if test_dataset in ['BC5CDR', 'NCBI_disease']:
                loss_main, main_prob, _, _, _ = model(input_ids=input_ids, attention_mask=mask, labels=targets, device=device)
                prob_lst.extend(main_prob)
                test_loss += loss_main.item()
                nb_test_steps += 1
                predicted_labels = torch.argmax(main_prob, dim=1)
                targets = targets.view(-1)
                new_labels, tmp_test_accuracy = find_new_labels_MM_BC5(id2label, label2id, mapping, targets, predicted_labels)
            else:
                targets = find_new_target(id2label, label2id, mapping, targets)
                loss_main, main_prob, _, _, _ = model(input_ids=input_ids, attention_mask=mask, labels=targets, device=device)
                prob_lst.extend(main_prob)
                test_loss += loss_main.item()
                nb_test_steps += 1
                predicted_labels = torch.argmax(main_prob, dim=1)
                targets = targets.view(-1)
                tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
                new_labels = [id2label_train[class_id.item()] for class_id in predicted_labels]

            test_accuracy += tmp_test_accuracy
            pred_lst.extend(new_labels)

    test_accuracy /= nb_test_steps
    return test_accuracy, pred_lst, prob_lst


def cross_eval(train_dataset, test_dataset, model, tokenizer, mapping_file, device):
    """Perform cross-evaluation between a training dataset and a test dataset.

    Args:
        train_dataset (str): Name of the training dataset.
        test_dataset (str): Name of the test dataset.
        model (torch.nn.Module): The trained model to use for evaluation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for processing text.
        mapping_file (str): Path to the file containing label mappings.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: A tuple containing:
            - float: The test accuracy.
            - list: A list of predicted labels.
            - list: A list of prediction probabilities.
    """
    print(f'\n\n\t\t\tCROSS TESTING\n\n')
    mapping = load_mapping(mapping_file)
    all_labels = read_train_labels(train_dataset)
    if test_dataset in ['BC5CDR', 'NCBI_disease']:
        _, label2id, test_data = load_data(test_dataset, 'test', tokenizer)
        id2label, _ = IdToLabelAndLabeltoId(all_labels)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, prob_lst = inference_MM_BC5(train_dataset, test_dataset, model, test_dataloader, tokenizer, device, id2label, label2id, mapping)
        return test_accuracy, pred_lst, prob_lst
    elif train_dataset == 'BC5CDR':
        id2label_test, _, test_data = load_data(test_dataset, 'test', tokenizer, cross_eval=True, mapping=mapping)
        id2label_train, label2id_train = IdToLabelAndLabeltoId(all_labels)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, prob_lst = inference_MM_BC5(train_dataset, test_dataset, model, test_dataloader, tokenizer, device, id2label_test, label2id_train, mapping, id2label_train)
        return test_accuracy, pred_lst, prob_lst
