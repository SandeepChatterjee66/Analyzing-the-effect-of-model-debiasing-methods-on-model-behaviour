"""
This script test the model trained for paraphrase identification task. It saves the prediction file and the softmax score file

Usage::

    python test.py --input_model_path <path_to_pretrained_model> --paws_file_path <path_to_paws_file>

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    --paws_file_path (str): Path to the PAWS file.
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
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from Generate_Embeddings.Paraphrase_identification.Vanilla.data_loader import *

input_path = './'
num_labels = 2
BATCH_SIZE = 32

class LossFunction(nn.Module):
    """
    Custom loss function class for the model.
    """
    def forward(self, probability):
        """
        Computes the negative log likelihood loss.
        
        Args:
            probability (torch.Tensor): The probabilities for the predictions.
        
        Returns:
            torch.Tensor: Computed loss.
        """
        loss = torch.log(probability)
        loss = -1 * loss
        loss = loss.mean()
        return loss


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



class MainModel(BertPreTrainedModel):
    """
    Main model class.
    """
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config (transformers.PretrainedConfig): Configuration for the model.
            loss_fn (custom, optional): Loss function to use. Default is None.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased",config = config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor): Input IDs of a batch.
            attention_mask (torch.Tensor): Attention mask of a batch.
            token_type_ids (torch.Tensor): Token type IDs.
            labels (torch.Tensor): Target labels of a batch.
            device (str): Device to run the model on(gpu or cpu).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loss and probabilities.
        """
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:]
        classifier_out = self.classifier(output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob


def inference(model, dataloader, tokenizer, device):
    """
    Perform inference on the provided data using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
        device (str): Device to perform computation ('cuda' or 'cpu').
        data (str): Dataset used for testing. Default is 'mnli'.

    Returns:
        tuple: Test accuracy, list of predicted labels, list of probabilities.
    """
    model.eval()
    prob_lst = []
    pred_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        with torch.no_grad():
            loss_main,main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
        test_loss += loss_main.item()
        nb_test_steps += 1
        prob_lst.extend(main_prob)
        pred = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        pred_lst.extend(pred)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), pred.cpu().numpy())
        test_accuracy += tmp_test_accuracy
        
    test_accuracy = test_accuracy / nb_test_steps
    
    return test_accuracy, pred_lst, prob_lst



def generate_prediction_file(pred_lst, output_file):
    """
    Generate a file containing predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file_path (str): Path to the output file.
    
    Returns:
        None
    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')

def generate_prob_file(prob_lst, prob_output_file_path):
    """
    Generate a file containing prediction probabilities.

    Args:
        prob_lst (list): List of probabilities for each prediction.
        output_file_path (str): Path to the output file.
    
    Returns:
        None
    """
    with open(prob_output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')



def main():
    """
    Main function to run the model training and inference pipeline.
    """
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--paws_file_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    model = MainModel.from_pretrained(args.input_model_path, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    
    data = read_dataset('../resources/QQP/train.pkl')
    train_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    train_accuracy, train_pred_lst, train_prob_lst = inference(model, train_dataloader, tokenizer, device)
    print(f'\tTrain dataset accuracy : {train_accuracy}')

    paws_data = load_paws(file_path='../resources/PAWS/test.tsv', tokenizer=tokenizer)
    paws_dataloader = DataLoader(paws_data, shuffle = False, batch_size=BATCH_SIZE)
    
    paws_test_accuracy, paws_pred_lst, paws_prob_lst = inference(model, paws_dataloader, tokenizer, device)
    print(f'\tPAWS test accuracy : {paws_test_accuracy}')
    
    data = read_dataset('../resources/QQP/val.pkl')
    val_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    val_accuracy, val_pred_lst, val_prob_lst = inference(model, val_dataloader, tokenizer, device)
    print(f'\tValidation dataset accuracy : {val_accuracy}')

    paws_output_file_path = os.path.join(input_path,'Predictions', 'PAWS', 'pred.txt')

    generate_prediction_file(paws_pred_lst, paws_output_file_path)
    generate_prob_file(paws_prob_lst, './Predictions/PAWS/prob_paws.txt')

    generate_prediction_file(train_pred_lst, './Predictions/train_pred.txt')
    generate_prob_file(train_prob_lst, './Predictions/train_prob.txt')
    
    generate_prediction_file(val_pred_lst, './Predictions/val_pred.txt')
    generate_prob_file(val_prob_lst, './Predictions/val_prob.txt')

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total testing time : {total_time}\n')

    print(f"Total testing time : {total_time}")
if __name__ == '__main__':
    main()
