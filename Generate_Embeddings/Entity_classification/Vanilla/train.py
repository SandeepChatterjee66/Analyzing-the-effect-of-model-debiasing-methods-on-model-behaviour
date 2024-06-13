"""
This script trains a BERT-based model for Entity Classification task using the Huggingface library.
It defines custom loss functions, data loaders, training, and validation functions, and supports model saving 
and early stopping.

Usage::

    python train.py --dataset_name <DATASET_NAME> --output_model_directory <OUTPUT_MODEL_DIR> --output_tokenizer_directory <OUTPUT_TOKENIZER_DIR>

Arguments::

    --dataset_name: Name of the dataset directory containing the 'questions.csv' file for training.
    --output_model_directory: Directory where the trained model will be saved.
    --output_tokenizer_directory: Directory where the tokenizer will be saved.
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
from Generate_Embeddings.Entity_classification.Vanilla.data_loader import load_data

# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
output_path = 'resources'
log_soft = F.log_softmax
tokenizer_dir = "./tokenizer"
model_dir = "./model"
config_dir = "./config"
print(torch.version.cuda)
MAX_LEN = 512 # suitable for all datasets
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
num_labels = 0


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



class MainModel(BertPreTrainedModel):
    """
    Main model class.
    """
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config (transformers.PretrainedConfig): Configuration for the model.
            loss_fn (optional): Loss function to use. Defaults to None.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1",config = config)
        self.hidden_layer = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels),self.num_labels)

    def forward(self, input_ids, attention_mask, labels,device):
        """
        Forward pass for the model.
        
        Args:
            input_ids (torch.Tensor): Input IDs of a batch.
            attention_mask (torch.Tensor): Attention mask of a batch.
            labels (torch.Tensor): Target labels of a batch.
            device (str): Device to run the model on(gpu or cpu).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Loss and probabilities.
        """
        output = self.bert(input_ids, attention_mask = attention_mask)
        output = output.last_hidden_state
        output = output[:,0,:]
        hidden_output = self.hidden_layer(output)
        classifier_out = self.classifier(hidden_output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob
        
def train(model, dataloader, optimizer, device):
    """
    Function to train the model for one epoch.
    
    Args:
        model (nn.Module): The model that is to be trained.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to run the model on.
        
    Returns:
        None
    """
    tr_loss, tr_accuracy = 0, 0
    bias_loss = 0
    nb_tr_steps = 0
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)

        tr_loss += loss_main.item()
        nb_tr_steps += 1
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        if idx % 100 == 0:
            print(f'\tModel loss at {idx} steps: {tr_loss}')
            if idx != 0:
                print(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tModel Loss at {idx} steps : {tr_loss}\n')
                if idx != 0:
                    fh.write(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()
        

    print(f'\tModel loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    """
    Validate the model.
    
    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run the model on.
    
    Returns:
        tuple: Validation loss and accuracy.
    """
    eval_loss = 0
    bias_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss : {eval_loss}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss, eval_accuracy/nb_eval_steps 



def main():
    """
    Main function to train and validate the model.
    """
    print("Training model :")
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    
    args = parser.parse_args()
    
    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory)
    
     
     
    with open('live.txt', 'a') as fh:
        fh.write(f'Dataset : {args.dataset_name}\n')
        fh.write(f'Model Path : {output_model_path}\n')
        
        
    best_output_model_path = output_model_path + '/BestModel'    
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)

    global num_labels
    if args.dataset_name == "BC5CDR":
        num_labels = 2
    elif args.dataset_name == "MedMentions":
        num_labels = 128
    else:
        num_labels = 1
    print(f'Dataset : {args.dataset_name}')
    print(f'Number of labels : {num_labels}')


    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    id2label, label2id, train_data,_ = load_data(args.dataset_name, 'train', tokenizer)
    # print(train_data[0])
    _,_,devel_data,_ = load_data(args.dataset_name, 'devel', tokenizer)
    config = AutoConfig.from_pretrained("dmis-lab/biobert-v1.1" , num_labels=num_labels)
    model = MainModel.from_pretrained("dmis-lab/biobert-v1.1", config = config , loss_fn = LossFunction())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    train_dataloader = DataLoader(train_data, shuffle = True, batch_size=BATCH_SIZE)
    devel_dataloader = DataLoader(devel_data, shuffle=True, batch_size=BATCH_SIZE)
    num_epochs = 20
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, devel_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')
        if eval_acc >= max_acc:
            max_acc = eval_acc
            patience = 0
            best_model = model
            best_tokenizer = tokenizer
        else:
            patience += 1
            if patience > 5:
                print("Early stopping at epoch : ",epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(best_output_model_path)
                patience = 0
                break

    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_tokenizer_path)
    
    best_model.save_pretrained(best_output_model_path)
    best_tokenizer.save_pretrained(best_output_model_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")

if __name__ == '__main__':
    main()