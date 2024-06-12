"""
This script trains a BERT-based model for Natural Language Inference(NLI) task using the Huggingface library.
It defines custom loss functions, data loaders, training, and validation functions, and supports model saving 
and early stopping.

Usage::

    python train.py --dataset_name <DATASET_NAME> --output_model_directory <OUTPUT_MODEL_DIR> --output_tokenizer_directory <OUTPUT_TOKENIZER_DIR>

Arguments::

    --dataset_name: Name of the dataset directory containing the 'questions.csv' file for training.
    --output_model_directory: Directory where the trained model will be saved.
    --output_tokenizer_directory: Directory where the tokenizer will be saved.
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
import pickle
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from NLI.generate_embeddings.Vanilla.data_loader import load_mnli
from NLI.generate_embeddings.Vanilla.data_loader import load_hans
from NLI.generate_embeddings.Vanilla.data_loader import load_snli
from torch.utils.data import random_split

MAX_LEN = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
input_path = './'

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
        # print(loss)
        loss = loss.mean()
        # print(loss)
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
            loss_fn (custom, optional): Loss function to use. Default is None.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = 3
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased",config = config)
        self.hidden = nn.Linear(768, 2*(self.num_labels))
        self.classifier = nn.Linear(2*(self.num_labels), self.num_labels)

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
        hidden_output = self.hidden(output)
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
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)

        tr_loss += loss_main.item()
        nb_tr_steps += 1
        #compute training accuracy
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
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy

    
    return eval_loss, eval_accuracy/nb_eval_steps 



def save_dataset(data, data_path):
    """
    Saves a dataset to a pickle file.
    
    Args:
        data: The dataset to save.
        data_path (str): Path to the pickle file to save the dataset.
        
    Returns:
        None
    """
    with open(data_path, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return

def main():
    """
    Main function to train and validate the model.
    
    Args:
        None

    Returns:
        None
    """
    gc.collect()
    
    torch.cuda.empty_cache()
    print("Training model :")
    start = time.time()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    
    args = parser.parse_args()

    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory)
 
    best_output_model_path = output_model_path + '/BestModel'    
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)

    train_file_path = os.path.join('../resources',args.dataset_name, args.dataset_name + '_train.txt')
    dev_matched_file_path = os.path.join('../resources', args.dataset_name, args.dataset_name + '_dev_matched.txt')
    dev_mismatched_file_path = os.path.join('../resources', args.dataset_name, args.dataset_name + '_dev_mismatched.txt')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = load_mnli(file_path=train_file_path, tokenizer=tokenizer)
    print(data[0])
    return
    
    model = MainModel.from_pretrained("bert-base-uncased", num_labels = 3, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    dev_data1 = load_mnli(file_path=dev_matched_file_path, tokenizer=tokenizer, type = False)
    dev_data2 = load_mnli(file_path=dev_mismatched_file_path, tokenizer=tokenizer,type = False)

    dev_size = int(0.5 * len(dev_data1))
    test_size = len(dev_data1) - dev_size
    dev_data1, test_data1 = random_split(dev_data1, [dev_size, test_size])

    dev_size = int(0.5 * len(dev_data2))
    test_size = len(dev_data2) - dev_size
    dev_data2, test_data2 = random_split(dev_data2, [dev_size, test_size])

    eval_data = ConcatDataset([dev_data1, dev_data2])
    test_data = ConcatDataset([test_data1, test_data2])
    
    print(f'Validation data length : {len(eval_data)}')
    save_dataset(eval_data, '../resources/multinli_1.0/val.pkl')
    save_dataset(test_data, '../resources/multinli_1.0/test.pkl')
    
    

    train_dataloader = DataLoader(data, shuffle = True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=BATCH_SIZE)

    num_epochs = 10
    max_acc = 0.0
    patience = 0
    max_hans_acc = 0.0
    max_mnli_mm_acc = 0.0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')
        print(f'\tValidation accuracy for epoch: {eval_acc}')
        with open('live.txt', 'a') as fh:
            fh.write(f'\tValidation Loss : {validation_loss}\n')
            fh.write(f'\tValidation accuracy for epoch: {eval_acc}\n')
        
        if eval_acc > max_acc:
            max_acc = eval_acc
            patience = 0
            best_model = model
            best_tokenizer = tokenizer
            best_model.save_pretrained(best_output_model_path)
            best_tokenizer.save_pretrained(best_output_model_path)
        else:
            patience += 1
            if patience > 2:
                print("Early stopping at epoch : ",epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(best_output_model_path)
                patience = 0
                break
        model.save_pretrained(os.path.join(input_path, args.output_model_directory,f'epoch{epoch}'))
        tokenizer.save_pretrained(os.path.join(input_path, args.output_model_directory,f'epoch{epoch}'))

    best_model.save_pretrained(best_output_model_path)
    best_tokenizer.save_pretrained(best_output_model_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
