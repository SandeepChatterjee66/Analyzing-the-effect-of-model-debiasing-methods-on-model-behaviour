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


import os 
import time
import math 
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification,BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
import torch.nn.functional as F
from seqeval.metrics import classification_report
import torch.nn as nn
from Entity_classification.generate_embeddings.Disentangled_feature_augmentation.train import MainModel
from Entity_classification.generate_embeddings.Disentangled_feature_augmentation.data_loader import load_data
from Entity_classification.generate_embeddings.Disentangled_feature_augmentation.cross_eval import cross_eval

input_path = './'
output_path = 'resources'


MAX_LEN = 512
BATCH_SIZE = 36
num_labels = 128
count_labels = []




def inference(model, dataloader, tokenizer, device, id2label):
    """Performs inference on the given model and dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the data to evaluate.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for data preprocessing.
        device (torch.device): Device to perform computations on (CPU or GPU).
        id2label (dict): Dictionary mapping label IDs to label names.

    Returns:
        tuple: Contains test accuracy, list of predicted labels, weights, and probabilities.
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
            z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask)
            z = torch.cat((z_l,z_b), dim = 1)
            pred_conflict ,pred_align = model.linear(z_conflict = z, z_align = z)
            pred_prob = F.softmax(pred_conflict, dim = 1)
            prob_lst.extend(pred_prob)
            predicted_labels = torch.argmax(pred_prob,dim=1)
            loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
            loss_dis_align = F.cross_entropy(pred_align,targets.view(-1), reduction = 'none')
            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
            loss = F.cross_entropy(pred_conflict, targets.view(-1))
            test_loss += loss
            targets = targets.view(-1)
            tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            labels = [id2label[class_id.item()] for class_id in predicted_labels]
            test_accuracy += tmp_test_accuracy
            pred_lst.extend(labels)
        weights.extend(loss_weight)
        # print(loss_weight)
    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst, weights, prob_lst


def generate_prediction_file(pred_lst, output_file):
    """Generates a prediction file from the list of predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file (str): Path to the output file where predictions will be written.

    Returns:
        None
    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')
                

def main():
    """Main function to run the evaluation process.

    This function performs the following steps:
    1. Parse command line arguments to get dataset names, model and tokenizer directories, mapping file, and output file.
    2. Determine the number of labels based on the training dataset name.
    3. Load the tokenizer and model from the specified directories.
    4. Perform inference on the test dataset.
    5. Generate prediction files for the predictions, weights, and probabilities.
    6. Log the total execution time.

    Args:
        --train_dataset_name (str): Name of the training dataset.
        --test_dataset_name (str): Name of the test dataset.
        --model_directory (str): Directory where the trained model is stored.
        --tokenizer_directory (str): Directory where the tokenizer is stored.
        --mapping_file (str, optional): Path to the mapping file.
        --output_file (str): Path to the output file for predictions.

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
    model = MainModel.from_pretrained(input_model_path, config = config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    if(args.train_dataset_name == args.test_dataset_name):
        id2label, label2id, test_data,_ = load_data(args.train_dataset_name, 'test', tokenizer)
        test_dataloader = DataLoader(test_data, shuffle = False, batch_size=BATCH_SIZE)
        test_accuracy, pred_lst, weights, prob_lst = inference(model, test_dataloader, tokenizer, device, id2label)
        print(f'\t{args.train_dataset_name} test accuracy: {test_accuracy}')
    else:
        test_accuracy, pred_lst, weights, prob_lst = cross_eval(args.train_dataset_name, args.test_dataset_name, model, tokenizer, args.mapping_file, device)
        print(f'\t{args.test_dataset_name} test accuracy on {args.train_dataset_name} model: {test_accuracy}')
    weights_path = args.output_file.split('/')
    weight_file_path = os.path.join(input_path, weights_path[0], weights_path[1], 'weights.txt')
    print(weight_file_path)
    generate_prediction_file(pred_lst, output_file_path)
    generate_prediction_file(weights, weight_file_path)


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
    
if __name__ == '__main__':
    main()
