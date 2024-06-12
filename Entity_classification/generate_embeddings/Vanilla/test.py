"""
This script test the model trained for Entity Classification task. It saves the prediction file and the softmax score file

Usage::

    python3 test.py --train_dataset_name <train_dataset_name> --test_dataset_name <test_dataset_name> --model_directory <trained_model_directory> --tokenizer_directory <saved_tokenizer_directory> --output_file <path_to_output_pred_file> --mapping_file <mapping_file>

Arguments::

    --train_dataset_name (str): Name of train dataset.
    --test_dataset_name (str): Name of test dataset.
    --model_directory (str): Path to the pretrained model directory.
    --tokenizer_directory (str): Path to the saved model directory.
    --output_file (str): Path to the output prediction file.
    --mapping_file (str): Path to the mapping file.
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
from Entity_classification.generate_embeddings.Vanilla.data_loader import * #load_data
from Entity_classification.generate_embeddings.Vanilla.train import * #MainModel as Model
from Entity_classification.generate_embeddings.Vanilla.train import * #LossFunction as LossFunction
from Entity_classification.generate_embeddings.Vanilla.cross_eval import * #cross_eval

input_path = './'
output_path = 'resources'


MAX_LEN = 512
BATCH_SIZE = 36
num_labels = 0


def inference(model, dataloader, tokenizer, device, id2label):
    """
    Perform inference on the provided data using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
        device (str): Device to perform computation ('cuda' or 'cpu').
        id2label (dict): Mapping from label ids to label names.

    Returns:
        tuple: Test accuracy, list of predicted labels, list of probabilities for each input data.
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
            loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets, device = device)
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


def generate_prediction_file(pred_lst, output_file_path):
    """
    Generate a file containing predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file_path (str): Path to the output file.
        
    Returns:
        None
    """
    with open(output_file_path, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')


def generate_prob_file(prob_lst, output_file_path):
    """
    Generate a file containing prediction probabilities.

    Args:
        prob_lst (list): List of probabilities for each prediction.
        output_file_path (str): Path to the output file.
        
    Returns:
        None
    """
    with open(output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')


def main():
    """
    Main function to run the model training and inference pipeline.
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
    model = Model.from_pretrained(input_model_path, config = config, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    if(args.train_dataset_name == args.test_dataset_name):
        id2label, label2id, train_data,_ = load_data(args.train_dataset_name, 'train', tokenizer)
        train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
        train_accuracy, train_pred_lst, train_prob_lst = inference(model, train_dataloader, tokenizer, device, id2label)
        print(f'\t{args.train_dataset_name} train accuracy: {train_accuracy}')
        generate_prediction_file(train_pred_lst, './Predictions/MM_pred.txt')
        generate_prob_file(train_prob_lst, './Predictions/MM_prob.txt')
        id2label, label2id, test_data,_ = load_data(args.train_dataset_name, 'test', tokenizer)

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

    generate_prob_file(prob_lst, prob_output_file_path)
    

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
