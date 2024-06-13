"""
This script trains a BERT-based model for Named Entity Recognition (NER)
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

from data_loader import load_data
#from Generate_Embeddings.Entity_classification.Feature_sieve.data_loader import load_data


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
MAX_GRAD_NORM = 10
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
alpha1 = 0.2
alpha2 = 2
alpha3 = 0.9
num_labels = 0


class BiasModel(nn.Module):
    """
    A class representing the Bias Model.

    Attributes:
        num_labels (int): Number of labels for classification.
    """
    def __init__(self,attention_size, hidden_size, num_labels):
        """
        Initializes the BiasModel.

        Args:
            attention_size (int): Size of the attention.
            hidden_size (int): Size of the hidden layer.
            num_labels (int): Number of labels for classification.
        """
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)  
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size,attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
    def forward(self, bert_hidden_layer_input):
        """
        Performs forward pass of the Bias Model.

        Args:
            bert_hidden_layer_input: Input tensor to the Bias Model.

        Returns:
            logits: Output logits of the Bias Model.
        """
        bert_hidden_layer_input = self.dropout(bert_hidden_layer_input)
        # print(bert_hidden_layer_input.shape)
        attention1_out = self.attention1(bert_hidden_layer_input)
        # print(attention1_out.shape)
        attention1_out = torch.tanh(attention1_out)
        # print(attention1_out.shape)
        attention2_out = self.attention2(attention1_out)
        # print(attention2_out.shape)
        # print(attention2_out)
        attention2_out = F.softmax(attention2_out, dim = 1)
        # print(attention2_out.shape)
        # print(attention2_out)
        # print(torch.sum(attention2_out, dim = 1))
        # print(attention2_out.unsqueeze(-1).shape)
        weighted_sum = torch.sum(attention2_out * bert_hidden_layer_input, dim=1)
        # print(weighted_sum.shape)
        logits = self.classifier(weighted_sum)
        # print(logits)
        return logits
        



class MainModel(BertPreTrainedModel):
    """
    A class representing the Main Model for Named Entity Recognition (NER).

    Attributes:
        num_labels (int): Number of labels for classification.
        loss_fn (callable): Loss function used for training.
    """
    def __init__(self, config, loss_fn = None):
        """
        Initializes the MainModel.

        Args:
            config: Configuration object.
            loss_fn (callable): Loss function used for training.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("dmis-lab/biobert-v1.1",config = config)
        self.bias_model = BiasModel(attention_size = 1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels), self.num_labels)

    def forward(self, input_ids, attention_mask, labels,device):
        """
        Performs forward pass of the Main Model.

        Args:
            input_ids: Input IDs tensor.
            attention_mask: Attention mask tensor.
            labels: Labels tensor.
            device: Device to run the model on.

        Returns:
            main_loss: Main model loss.
            main_pred: Main model predictions.
            bias_loss: Bias model loss.
            bias_pred: Bias model predictions.
            forget_loss: Forget model loss.
        """  
        output = self.bert(input_ids, attention_mask = attention_mask)
        hidden_state = output.hidden_states[4]
        hidden_state_detached = hidden_state.detach()
        bias_prob = self.bias_model(hidden_state_detached)
        bias_loss = F.cross_entropy(bias_prob, labels.view(-1))
        forget_prob = self.bias_model(hidden_state)
        pseudo_labels = torch.ones_like(forget_prob) / self.num_labels
        forget_loss =  F.cross_entropy(forget_prob, pseudo_labels)
        # print(hidden_state.shape)
        # output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:]
        hidden_output = self.hidden_layer(output)
        main_prob = self.classifier(hidden_output)
        main_loss = F.cross_entropy(main_prob, labels.view(-1))
        main_pred = F.softmax(main_prob, dim = 1)
        bias_pred = F.softmax(bias_prob, dim = 1)
        # print("loss")
        # print(loss_main)
        # print(loss_bias)
        return main_loss, main_pred, bias_loss, bias_pred, forget_loss
    
    def set_bias_grad(self, requires_grad):
        """
        Sets the gradient requirement for bias model parameters.

        Args:
            requires_grad (bool): Whether to require gradients or not.
        """
        for param in self.bias_model.parameters():
            param.requires_grad = requires_grad
    
        
def train(model, dataloader, optimizer, device):
    """
    Trains the BERT-based model for Named Entity Recognition (NER).

    Args:
        model (MainModel): The BERT-based model for NER.
        dataloader (DataLoader): DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (str): The device to use for training ('cuda' or 'cpu').
    
    Returns:
        None
    """
    tr_loss, tr_accuracy = 0, 0
    total_bias_loss = 0
    total_forget_loss = 0
    tr_accuracy_bias = 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        main_loss, main_pred, bias_loss, bias_pred, forget_loss = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)

        # print(f'\tLoss Main : {loss_main}')
        tr_loss += main_loss.item()
        total_bias_loss += bias_loss.item()
        total_forget_loss += forget_loss.item()
        nb_tr_steps += 1
        #compute training accuracy
        predicted_labels_main = torch.argmax(main_pred, dim=1)
        predicted_labels_bias = torch.argmax(bias_pred, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels_main.cpu().numpy())
        tmp_tr_accuracy_bias = accuracy_score(targets.cpu().numpy(), predicted_labels_bias.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        tr_accuracy_bias += tmp_tr_accuracy_bias
        if idx % 100 == 0:
            print(f'\tMain Model loss at {idx} steps: {tr_loss}')
            print(f'\tBias Model loss at {idx} steps: {total_bias_loss}')
            print(f'\tForget Model loss at {idx} steps: {total_forget_loss}')
            if idx != 0:
                print(f'\tMain Model Accuracy : {tr_accuracy/nb_tr_steps}')
                print(f'\tBias Model Accuracy : {tr_accuracy_bias/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tMain Model Loss at {idx} steps : {tr_loss}\n')
                fh.write(f'\tBias Model Loss at {idx} steps : {total_bias_loss}\n')
                fh.write(f'\tForget Model Loss at {idx} steps : {total_forget_loss}\n')
                if idx != 0:
                    fh.write(f'\tMain Model Accuracy : {tr_accuracy/nb_tr_steps}')
                    fh.write(f'\tBias Model Accuracy : {tr_accuracy_bias/nb_tr_steps}')
        # print(f'Main loss : {loss_main} Bias Loss : {loss_bias} Accuracy : {tmp_tr_accuracy}')
        # print(tmp_tr_accuracy)
        # bias_model_state_dict = model.bias_model.state_dict()
        # print(model.bias_model.state_dict())
        # print("                    2nd Output                  ")
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )
        # print(model.bert.encoder.layer[0].state_dict())
        optimizer.zero_grad()
        for param in model.bias_model.parameters():
            param.requires_grad = False
        torch.autograd.set_detect_anomaly(True)
        loss = alpha1 * main_loss
        # print(idx)
        if idx % 2 == 0:
            loss += alpha3 * forget_loss
        loss.backward(retain_graph = True)
        optimizer.step()
        for param in model.bias_model.parameters():
            param.requires_grad = True
        # print("\n\n\n\nOutput 2 \n\n\n\n")
        # print(model.bert.encoder.layer[0].state_dict())
        # model.bias_model.load_state_dict(bias_model_state_dict)
        # print(" \n\n                   3rd Output                  ")
        # print(model.bias_model.state_dict())

        optimizer.zero_grad()
        bias_loss = alpha2 * bias_loss
        bias_loss.backward()
        optimizer.step()
        # print(" \n\n                   3rd Output                  ")
        # print(model.bias_model.state_dict())

    print(f'\tMain Model loss for the epoch: {tr_loss}')
    print(f'\tBias Model loss for the epoch: {bias_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    """
    Validates the performance of the BERT-based model for Named Entity Recognition (NER).

    Args:
        model (MainModel): The BERT-based model for NER.
        dataloader (DataLoader): DataLoader containing the validation data.
        device (str): The device to use for validation ('cuda' or 'cpu').
    
    Returns:
        tuple: A tuple containing the evaluation loss and evaluation accuracy.
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

        loss_main, main_prob, loss_bias,_,_ = model(input_ids=input_ids, attention_mask=mask, labels=targets,device = device)
        bias_loss += loss_bias.item()
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss : {eval_loss}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss, eval_accuracy/nb_eval_steps 

    

def main():
    """
    Main function for training the BERT-based model for Named Entity Recognition (NER).

    Args:
        None
    
    Returns:
        None
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
    id2label, label2id, train_data = load_data(args.dataset_name, 'train', tokenizer)
    # print(train_data[0])
    # print(len(label2id))
    _,_,devel_data = load_data(args.dataset_name, 'devel', tokenizer)
    config = AutoConfig.from_pretrained("dmis-lab/biobert-v1.1" , num_labels=num_labels)
    model = MainModel.from_pretrained("dmis-lab/biobert-v1.1", config = config, loss_fn = None)
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