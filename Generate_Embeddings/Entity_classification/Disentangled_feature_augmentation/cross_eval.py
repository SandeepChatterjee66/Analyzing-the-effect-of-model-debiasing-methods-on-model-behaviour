"""
This module provides functions for reading training labels, finding new labels and targets,
performing inference, and conducting cross-evaluation for named entity recognition tasks 
in biomedical datasets.

"""

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
from Generate_Embeddings.Entity_classification.Disentangled_feature_augmentation.train import *
from Generate_Embeddings.Entity_classification.Disentangled_feature_augmentation.data_loader import *

BATCH_SIZE = 36
input_path = './'






def read_train_labels(dataset_name):
    """
    Reads training labels from the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to read labels from.

    Returns:
        list: A list of labels read from the dataset.
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
    """
    Maps predicted labels to new labels based on a given mapping and computes accuracy.

    Args:
        id2label (dict): Mapping from ID to label.
        label2id (dict): Mapping from label to ID.
        mapping (dict): Mapping from one label set to another.
        targets (torch.Tensor): Ground truth target labels.
        predicted_labels (torch.Tensor): Predicted labels.

    Returns:
        tuple: A tuple containing a list of new labels and the test accuracy.
    """
    new_labels = []
    labels = [id2label[class_id.item()] for class_id in predicted_labels]
    for i, label in enumerate(labels):
        if mapping.get(label) is not None:
            new_labels.append(mapping[label])
        else:
            new_labels.append('nan')
    predicted_labels = []
    for i, label in enumerate(new_labels):
        if label == 'nan':
            predicted_labels.append(2)
        else:
            predicted_labels.append(label2id[label])
    tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), np.array(predicted_labels))
    return new_labels, tmp_test_accuracy


def find_new_target(id2label, label2id, mapping, targets):
    """
    Converts targets to a new format based on the provided mapping.

    Args:
        id2label (dict): Mapping from ID to label.
        label2id (dict): Mapping from label to ID.
        mapping (dict): Mapping from one label set to another.
        targets (torch.Tensor): Ground truth target labels.

    Returns:
        torch.Tensor: A tensor of new target labels.
    """
    new_targets = []
    targets = [id2label[target.item()] for target in targets]
    new_labels = [mapping[target] for target in targets]
    new_targets = [label2id[label] for label in new_labels]
    new_targets = torch.tensor(new_targets)
    device = 'cuda' if cuda.is_available() else 'cpu'
    new_targets = new_targets.to(device, dtype=torch.long)
    return new_targets.unsqueeze(1)


def inference_MM_BC5(train_dataset, test_dataset, model, dataloader, tokenizer, device, id2label, label2id, mapping, id2label_train=None):
    """
    Performs inference on the given dataset and returns accuracy, predicted labels, weights, and probabilities.

    Args:
        train_dataset (str): Name of the training dataset.
        test_dataset (str): Name of the testing dataset.
        model (torch.nn.Module): The trained model to use for inference.
        dataloader (DataLoader): DataLoader for the test dataset.
        tokenizer (transformer tokenizer object): Tokenizer for processing text.
        device (str): Device to perform computations on.
        id2label (dict): Mapping from ID to label.
        label2id (dict): Mapping from label to ID.
        mapping (dict): Mapping from one label set to another.
        id2label_train (dict, optional): Mapping from ID to label for training set. Defaults to None.

    Returns:
        tuple: A tuple containing test accuracy, predicted labels, weights, and probabilities.
    """
    model.eval()
    pred_lst = []
    weights = []
    prob_lst = []
    test_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        nb_test_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask)
            z = torch.cat((z_l, z_b), dim=1)
            pred_conflict, pred_align = model.linear(z_conflict=z, z_align=z)
            pred_prob = F.softmax(pred_conflict, dim=1)
            prob_lst.extend(pred_prob)
            predicted_labels = torch.argmax(pred_prob, dim=1)
            targets = targets.view(-1)
            if train_dataset == 'MedMentions' or test_dataset == 'NCBI_disease':
                loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
                loss_dis_align = F.cross_entropy(pred_align, targets.view(-1), reduction='none')
                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
                new_labels, tmp_test_accuracy = find_new_labels_MM_BC5(id2label, label2id, mapping, targets, predicted_labels)
            else:
                targets = find_new_target(id2label, label2id, mapping, targets)
                loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
                loss_dis_align = F.cross_entropy(pred_align, targets.view(-1), reduction='none')
                loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
                targets = targets.view(-1)
                tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
                new_labels = [id2label_train[class_id.item()] for class_id in predicted_labels]
            weights.extend(loss_weight)
            test_accuracy += tmp_test_accuracy
            pred_lst.extend(new_labels)

    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst, weights, prob_lst


def cross_eval(train_dataset, test_dataset, model, tokenizer, mapping_file, device):
    """
    Conducts cross-evaluation between training and testing datasets and returns evaluation metrics.

    Args:
        train_dataset (str): Name of the training dataset.
        test_dataset (str): Name of the testing dataset.
        model (torch.nn.Module): The trained model to use for evaluation.
        tokenizer (transformers.AutoTokenizer): Tokenizer for processing text.
        mapping_file (str): Path to the mapping file.
        device (str): Device to perform computations on.

    Returns:
        tuple: A tuple containing test accuracy, predicted labels, weights, and probabilities.
    """
    print('\n\n\t\t\tCROSS TESTING\n\n')
    mapping = load_mapping(mapping_file)
    all_labels = read_train_labels(train_dataset)
    if train_dataset == 'MedMentions' or test_dataset == 'NCBI_disease':
        _, label2id, test_data, _ = load_data(test_dataset, 'test', tokenizer)
        id2label, _ = IdToLabelAndLabeltoId(all_labels)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, weights, prob_lst = inference_MM_BC5(train_dataset, test_dataset, model, test_dataloader, tokenizer, device, id2label, label2id, mapping)
        return test_accuracy, pred_lst, weights, prob_lst
    elif train_dataset == 'BC5CDR':
        id2label_test, _, test_data, _ = load_data(test_dataset, 'test', tokenizer, cross_eval=True, mapping=mapping)
        id2label_train, label2id_train = IdToLabelAndLabeltoId(all_labels)
        test_dataloader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, weights, prob_lst = inference_MM_BC5(train_dataset, test_dataset, model, test_dataloader, tokenizer, device, id2label_test, label2id_train, mapping, id2label_train)
        return test_accuracy, pred_lst, weights, prob_lst
