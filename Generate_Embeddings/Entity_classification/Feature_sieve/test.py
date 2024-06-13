"""
This script is designed to perform inference using a pre-trained transformer model for token classification tasks. The script evaluates the model on test datasets, either within the same dataset or across different datasets, and generates prediction files, weights, and probability distributions for the predicted labels.

Usage::

    python <path_to_script> --train_dataset_name <train_dataset> --test_dataset_name <test_dataset> --model_directory <model_directory> --tokenizer_directory <tokenizer_directory> --mapping_file <mapping_file> --output_file <output_file>

Arguments::

    --train_dataset_name ( str ) : The name of the training dataset (e.g., 'BC5CDR', 'MedMentions').
    --test_dataset_name ( str ) : The name of the test dataset (e.g., 'BC5CDR', 'MedMentions').
    --model_directory ( str ) : The directory containing the pre-trained model.
    --tokenizer_directory ( str ) : The directory containing the tokenizer.
    --mapping_file ( str ) : The file path for the mapping file used in cross-dataset evaluation (optional).
    --output_file ( str ) : The file path for the output prediction file.

Details::
    - The script initializes the tokenizer and model using the directories provided.
    - It determines the number of labels based on the training dataset.
    - Depending on whether the training and test datasets are the same or different, it performs inference using either a standard evaluation or a cross-evaluation.
    - The results are saved in specified output files, including predictions, weights, and probabilities.
"""

from multiprocessing import reduction
import pandas as pd
import time
from tqdm import tqdm
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

from data_loader import load_data
from train import MainModel as Model
from cross_eval import cross_eval

# from Generate_Embeddings.Entity_classification.Feature_sieve.data_loader import load_data
# from Generate_Embeddings.Entity_classification.Feature_sieve.train import MainModel as Model
# from Generate_Embeddings.Entity_classification.Feature_sieve.cross_eval import cross_eval

input_path = './'
output_path = 'resources'


MAX_LEN = 512
BATCH_SIZE = 36
num_labels = 128


def inference(model, dataloader, tokenizer, device, id2label):
    """
    Performs inference using the provided model.

    Args:
        model (MainModel): Main model for inference.
        dataloader (DataLoader): DataLoader for loading data.
        tokenizer (AutoTokenizer): Tokenizer for tokenizing input.
        device (str): Device to run the model on.
        id2label (dict): Mapping from label index to label.

    Returns:
        float: Test accuracy.
        list: List of predicted labels.
        list: List of probabilities.
    """
    model.eval()
    pred_lst = []
    prob_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        with torch.no_grad():
            loss_main, main_prob, _,_,_ = model(input_ids=input_ids, attention_mask=mask, labels=targets, device = device)
        test_loss += loss_main.item()
        prob_lst.extend(main_prob)
        nb_test_steps += 1
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        labels = [id2label[class_id.item()] for class_id in predicted_labels]
        test_accuracy += tmp_test_accuracy
        pred_lst.extend(labels)
        
    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst, prob_lst


def generate_prediction_file(pred_lst, output_file):
    """
    Generates a prediction file from a list of predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file (str): Output file path.

    Returns:
        None
    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')


def main():
    """
    Main function to run the inference process.

    This function performs the following steps:
    1. Parses command line arguments to get dataset names, model and tokenizer directories, mapping file, and output file.
    2. Initializes the main model, tokenizer, and device.
    3. Performs inference on the test dataset.
    4. Generates prediction file and probability file.
    5. Logs total inference time.

    Args:
        --train_dataset_name (str): Name of the training dataset.
        --test_dataset_name (str): Name of the test dataset.
        --model_directory (str): Directory containing the model.
        --tokenizer_directory (str): Directory containing the tokenizer.
        --mapping_file (str): File path for the mapping file (optional).
        --output_file (str): Output file path.

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
    model = Model.from_pretrained(input_model_path, config = config, loss_fn = None)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    if(args.train_dataset_name == args.test_dataset_name):
        id2label, label2id, test_data = load_data(args.train_dataset_name, 'test', tokenizer)
        test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, prob_lst = inference(model, test_dataloader, tokenizer, device, id2label)
        print(f'\t{args.train_dataset_name} test accuracy: {test_accuracy}')
    else:
        test_accuracy, pred_lst, prob_lst = cross_eval(args.train_dataset_name, args.test_dataset_name, model, tokenizer, args.mapping_file, device)
        print(f'\t{args.test_dataset_name} test accuracy on {args.train_dataset_name} model: {test_accuracy}')

    generate_prediction_file(pred_lst, output_file_path)

    prob_path = args.output_file.split('/')
    prob_output_file_path = os.path.join(input_path, prob_path[0], prob_path[1], 'prob.txt')
    print(prob_output_file_path)
    
    with open(prob_output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
