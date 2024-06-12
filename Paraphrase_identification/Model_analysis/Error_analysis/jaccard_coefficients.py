"""
This script performs evaluation of named entity recognition (NER) models by comparing predicted labels against true labels in biomedical datasets. The script includes functionalities for text preprocessing, handling cross-dataset evaluation, and generating detailed evaluation metrics such as precision, recall, F1-score, and accuracy.

Usage::

    python <path_to_script> --train_dataset_name <train_dataset> --test_dataset_name <test_dataset> --cui_dictionary_train <cui_dictionary_train> --cui_dictionary_test <cui_dictionary_test> --predictions <predictions> --mapping_file <mapping_file>

Arguments::

    --train_dataset_name ( str ) : The name of the training dataset (e.g., 'BC5CDR', 'MedMentions').
    --test_dataset_name ( str ) : The name of the test dataset (e.g., 'BC5CDR', 'MedMentions').
    --cui_dictionary_train ( str ) : The file path for the training dataset CUI dictionary.
    --cui_dictionary_test ( str ) : The file path for the test dataset CUI dictionary.
    --predictions ( str ) : The file path for the predicted labels.
    --mapping_file ( str ) : The file path for the mapping file used in cross-dataset evaluation (optional).
"""

from multiprocessing import reduction
import pandas as pd
import time
import numpy as np
import csv
import argparse
import math

def read_file(filename):
    """Reads lines from a file and returns a list of lines.
    
    Args:
        filename (str): The path to the file to be read.
        
    Returns:
        list: A list containing lines read from the file.
    """
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            key = line.rstrip()
            lines.append(key)
    return lines

def find_correct_outputs(model_predictions, ground_truth, all_entities):
    """Finds correct outputs from model predictions and ground truth.
    
    Args:
        model_predictions (list): A list containing model predictions.
        ground_truth (list): A list containing ground truth labels.
        all_entities (list): A list containing all entities.
        
    Returns:
        list: A list containing correct outputs.
    """
    correct_outputs = []
    for i, prediction in enumerate(model_predictions):
        if prediction.strip() != ground_truth[i].strip():
            correct_outputs.append(all_entities[i].strip() + " " + str(i))
    return correct_outputs

def calculate_iou(list1, list2):
    """Calculates Intersection over Union (IoU) of two lists.
    
    Args:
        list1 (list): The first list.
        list2 (list): The second list.
        
    Returns:
        float: The IoU value.
    """
    intersection = len(set(list1).intersection(set(list2)))
    union = len(set(list1).union(set(list2)))
    
    if union == 0:
        return 0.0
    else:
        return intersection / union

def load_mapping(mapping_file):
    """Loads mapping from a file into a dictionary.
    
    Args:
        mapping_file (str): The path to the mapping file.
        
    Returns:
        dict: A dictionary containing the mapping.
    """
    mapping = {}
    with open(mapping_file, 'r') as fh:
        for line in fh:
            line = line.split('\t')
            mapping[line[0].strip('\n')] = line[1].strip('\n')
    return mapping

def filter_entities(all_entities, ground_truth, mapping):
    """Filters entities based on a mapping.
    
    Args:
        all_entities (list): A list containing all entities.
        ground_truth (list): A list containing ground truth labels.
        mapping (dict): A dictionary containing the mapping.
        
    Returns:
        tuple: A tuple containing filtered ground truth and entities.
    """
    new_entities = []
    new_ground_truth = []
    for i,label in enumerate(ground_truth):
        if mapping.get(str(label.strip('\n'))) is not None:
            new_entities.append(all_entities[i])
            new_ground_truth.append(mapping[label])
    return new_ground_truth, new_entities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate IoU of correct outputs from two models.")
    parser.add_argument("--train_dataset_name")
    parser.add_argument("--test_dataset_name")
    parser.add_argument("--all_entities_file", help="Path to all_entities.txt file")
    parser.add_argument("--model1_pred_file", help="Path to Model1/pred.txt file")
    parser.add_argument("--model2_pred_file", help="Path to Model2/pred.txt file")
    parser.add_argument("--groundtruth_file", help="Path to groundtruth.txt file")
    parser.add_argument("--mapping_file", required=False)
    
    args = parser.parse_args()

    all_entities = read_file(args.all_entities_file)
    model1_predictions = read_file(args.model1_pred_file)
    model2_predictions = read_file(args.model2_pred_file)
    ground_truth = read_file(args.groundtruth_file)
    if(args.train_dataset_name == 'BC5CDR' and args.test_dataset_name == 'MedMentions'):
        mapping = load_mapping(args.mapping_file)
        ground_truth, all_entities = filter_entities(all_entities, ground_truth, mapping)

    print(len(model1_predictions))
    print(len(model2_predictions))
    print(len(ground_truth))
    print(len(all_entities))
    model1_correct_outputs = find_correct_outputs(model1_predictions, ground_truth, all_entities)
    model2_correct_outputs = find_correct_outputs(model2_predictions, ground_truth, all_entities)

    iou = calculate_iou(model1_correct_outputs, model2_correct_outputs)
    print("Intersection over Union (IoU):", iou)
