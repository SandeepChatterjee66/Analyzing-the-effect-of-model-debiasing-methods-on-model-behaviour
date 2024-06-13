"""
This script trains a BERT-based model for paraphrase identification task
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
from sklearn.metrics import accuracy_score # type: ignore
import torch
from torch.utils.data import Dataset, DataLoader # type: ignore
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel # type: ignore
from torch import cuda
from seqeval.metrics import classification_report # type: ignore
from config import Config as config
import os
import gc
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from Generate_Embeddings.Paraphrase_identification.Disentangled_feature_augmentation.util import EMA
from Generate_Embeddings.Paraphrase_identification.Disentangled_feature_augmentation.data_loader import load_qqp, load_paws
from torch.utils.data import random_split


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
num_labels = 3
# num_labels = 2

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
lambda_dis_align = 1
lambda_swap_align = 0
lambda_swap = 0.1
weight_decay = 0.001
step_size = 30
gamma = 0.6



class GeneralizedCELoss(nn.Module):
    """Generalized Cross-Entropy Loss.

    Attributes:
        q (float): The parameter q to control the weighting.
    """

    def __init__(self, q=0.7):
        """Initializes GeneralizedCELoss with parameter q.

        Args:
            q (float): The parameter q to control the weighting.
        """
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        """Forward pass for the loss computation.

        Args:
            logits (torch.Tensor): The predicted logits from the model.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss.
        """
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss_weight = (Yg.squeeze().detach()**self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class MainModel(BertPreTrainedModel):
    """Main model for classification using a dual BERT architecture.

    Attributes:
        num_labels (int): Number of labels for classification.
        model_l (transformers.AutoModel): First BERT model.
        model_b (transformers.AutoModel): Second BERT model.
        fc_l_1 (torch.nn.Linear): First linear layer for the first BERT model.
        fc_l_2 (torch.nn.Linear): Second linear layer for the first BERT model.
        fc_b_1 (torch.nn.Linear): First linear layer for the second BERT model.
        fc_b_2 (torch.nn.Linear): Second linear layer for the second BERT model.
    """

    def __init__(self, config):
        """Initializes the MainModel with BERT and linear layers.

        Args:
            config (transformers.PretrainedConfig): Configuration for the BERT model.
        """
        super(MainModel, self).__init__(config)
        self.num_labels = 2
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2 * self.num_labels)
        self.fc_l_2 = nn.Linear(2 * self.num_labels, self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2 * self.num_labels)
        self.fc_b_2 = nn.Linear(2 * self.num_labels, self.num_labels)

    def features(self, input_ids, attention_mask):
        """Extracts features from the input using both BERT models.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tokens.

        Returns:
            torch.Tensor, torch.Tensor: Features from the first and second BERT models.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask)
        z_b = self.model_b(input_ids, attention_mask=attention_mask)
        return z_l.last_hidden_state[:, 0, :], z_b.last_hidden_state[:, 0, :]

    def linear(self, z_conflict, z_align):
        """Passes the features through linear layers to get predictions.

        Args:
            z_conflict (torch.Tensor): Concatenated features for conflict resolution.
            z_align (torch.Tensor): Concatenated features for alignment.

        Returns:
            torch.Tensor, torch.Tensor: Predictions from conflict and alignment branches.
        """
        hidden_output1 = self.fc_l_1(z_conflict)
        output1 = self.fc_l_2(hidden_output1)
        hidden_output2 = self.fc_b_1(z_align)
        output2 = self.fc_b_2(hidden_output2)
        return output1, output2

def read_dataset(data_path):
    """Reads a dataset from a pickle file.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        object: Loaded dataset.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data

def load_dataset(file_path):
    """Loads a dataset from a pickle file.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        object: Loaded dataset.
    """
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def valid(model, dataloader, device):
    """Evaluates the model on a validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (str): Device to perform computation on (e.g., 'cuda' or 'cpu').

    Returns:
        float, float: Validation loss and accuracy.
    """
    eval_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        nb_eval_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask)
            z = torch.cat((z_l, z_b), dim=1)
            pred, _ = model.linear(z_conflict=z, z_align=z)
            loss = F.cross_entropy(pred, targets.view(-1))
            eval_loss += loss
            pred = F.softmax(pred, dim=1)
            predicted_labels = torch.argmax(pred, dim=1)
            targets = targets.view(-1)
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss : {eval_loss}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss / nb_eval_steps, eval_accuracy / nb_eval_steps

def train(model, dataloader, optimizer, device, swap, scheduler, sample_loss_ema_d, sample_loss_ema_b, dataset_name):
    """Trains the model on the given dataset.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to perform computation on (e.g., 'cuda' or 'cpu').
        swap (bool): Flag to indicate if feature swapping should be used.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        sample_loss_ema_d (Entity_classification.generate_embeddings.Disentangled_feature_augmentation.util.EMA): EMA for conflict loss.
        sample_loss_ema_b (Entity_classification.generate_embeddings.Disentangled_feature_augmentation.util.EMA): EMA for alignment loss.
        dataset_name (str): Name of the dataset being used.

    Returns:
        None
    """
    tr_loss, total_tr_accuracy_main, total_tr_accuracy_main_swap = 0, 0, 0
    total_tr_accuracy_bias, total_tr_accuracy_bias_swap = 0, 0
    bias_loss = 0
    nb_tr_steps = 0
    bias_criterion = GeneralizedCELoss(q=0.7)
    model.train()
    last_index = 0

    for idx, batch in enumerate(dataloader):
        index = batch['index']

        original_indices = batch['index']
        index_mapping = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(original_indices)}

        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        outputs = model.features(input_ids=input_ids, attention_mask=mask)
        z_l, z_b = outputs

        z = torch.cat((z_l, z_b), dim=1)
        pred, _ = model.linear(z_conflict=z, z_align=z)

        # Compute loss for conflict branch
        loss = F.cross_entropy(pred, targets.view(-1))

        sample_loss_ema_d.update(original_indices, loss)  # update EMA with conflict loss
        loss = sample_loss_ema_d.get(original_indices)  # get updated EMA loss

        pred_softmax = F.softmax(pred, dim=1)
        predicted_labels = torch.argmax(pred_softmax, dim=1)
        targets = targets.view(-1)
        total_tr_accuracy_main += accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())

        if swap:
            z = torch.cat((z_b, z_l), dim=1)
            swap_pred, _ = model.linear(z_conflict=z, z_align=z)

            swap_loss = F.cross_entropy(swap_pred, targets.view(-1))

            sample_loss_ema_d.update(original_indices, swap_loss)  # update EMA with swap loss
            swap_loss = sample_loss_ema_d.get(original_indices)  # get updated EMA swap loss

            swap_pred_softmax = F.softmax(swap_pred, dim=1)
            swap_predicted_labels = torch.argmax(swap_pred_softmax, dim=1)
            total_tr_accuracy_main_swap += accuracy_score(targets.cpu().numpy(), swap_predicted_labels.cpu().numpy())

            bias_loss = bias_criterion(pred, targets.view(-1))
            sample_loss_ema_b.update(original_indices, bias_loss)  # update EMA with bias loss
            bias_loss = sample_loss_ema_b.get(original_indices)  # get updated EMA bias loss

            total_loss = loss.mean() + lambda_swap * swap_loss.mean() + lambda_dis_align * bias_loss.mean()
            bias_outputs = sample_loss_ema_b.get(original_indices)
            bias_outputs = bias_outputs.mean(dim=0)

            bias_outputs_softmax = F.softmax(bias_outputs, dim=0)
            bias_predicted_labels = torch.argmax(bias_outputs_softmax, dim=0)
            total_tr_accuracy_bias += accuracy_score(targets.cpu().numpy(), bias_predicted_labels.cpu().numpy())

            total_tr_accuracy_bias_swap += accuracy_score(targets.cpu().numpy(), swap_predicted_labels.cpu().numpy())

            total_loss += lambda_swap_align * bias_outputs.mean()

            bias_loss += bias_criterion(swap_pred, targets.view(-1))
            total_loss += lambda_swap_align * bias_loss.mean()

            optimizer.zero_grad()
            total_loss.backward()

        else:
            optimizer.zero_grad()
            loss.mean().backward()

        optimizer.step()
        scheduler.step()

        tr_loss += loss.mean().item()
        nb_tr_steps += 1

    avg_loss = tr_loss / nb_tr_steps
    avg_accuracy_main = total_tr_accuracy_main / nb_tr_steps
    avg_accuracy_main_swap = total_tr_accuracy_main_swap / nb_tr_steps
    avg_accuracy_bias = total_tr_accuracy_bias / nb_tr_steps
    avg_accuracy_bias_swap = total_tr_accuracy_bias_swap / nb_tr_steps

    print(f"Average Loss: {avg_loss}")
    print(f"Average Main Accuracy: {avg_accuracy_main}")
    print(f"Average Main Swap Accuracy: {avg_accuracy_main_swap}")
    print(f"Average Bias Accuracy: {avg_accuracy_bias}")
    print(f"Average Bias Swap Accuracy: {avg_accuracy_bias_swap}")


'''
The `main` function serves as the entry point for training a model.
 Here's a breakdown of what it does:

1. Sets up necessary configurations and parameters for training,
 including command-line arguments parsing.
2. Creates directories for saving model checkpoints and logs.
3. Loads data and tokenizer, initializes the model, optimizer, and
 scheduler.
4. Defines data loaders for training and evaluation datasets.
5. Trains the model for a specified number of epochs, evaluating on
 validation and external datasets periodically.
6. Saves the best performing models based on validation and external
 dataset accuracy.
7. Implements early stopping based on a patience criterion.
8. Logs training progress and timing information.
9. Finally, saves the trained model and tokenizer to specified
 directories.

The `main` function orchestrates the entire training process, 
including data loading, model initialization, training loop execution,
 evaluation, and model saving.

'''

def main():
    """
    Main function to orchestrate the training process.

    Parses command-line arguments, sets up configurations, initializes model, tokenizer,
    optimizer, and scheduler. Loads data, performs training, evaluates on validation data,
    saves the best performing models, implements early stopping, and logs training progress.

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

    # train_file_path = os.path.join(input_path, args.dataset_name,'questions.csv')
    # dev_file_path= os.path.join(input_path,'PAWS/dev.tsv')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # data = load_qqp2(file_path=train_file_path, tokenizer=tokenizer)
    # print(data.len)
    

    config = AutoConfig.from_pretrained("bert-base-uncased" , num_labels=num_labels)
    model = MainModel.from_pretrained("bert-base-uncased", config = config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #Decays the learning rate of each parameter group by gamma every step_size epochs.

    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    data = read_dataset('../resources/QQP/train.pkl')
    dev_data = read_dataset('../resources/QQP/val.pkl')
    ### val_size = int(0.1 * len(data))
    ### train_size = len(data) - val_size
    #val_data = load_paws(file_path='./PAWS/test.tsv', tokenizer=tokenizer)
    val_size = int(len(data))
    train_size = int(len(data))
    print(f'train_size {(train_size)}')
    print(f'val_size {(val_size)}')

    #paws_data = load_paws(file_path='./PAWS/test.tsv', tokenizer=tokenizer)
    #paws_dataloader = DataLoader(paws_data, shuffle = True, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(data, shuffle = True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dev_data, shuffle=True, batch_size=BATCH_SIZE)

    lables_file = '../resources/QQP/labels_clean.txt'
    with open(lables_file, 'r') as infile:
        labels = [int(line.strip()) for line in infile]
        print(f'labels length {len(labels)}')

    sample_loss_ema_b = EMA(torch.LongTensor(labels), num_classes=num_labels)
    sample_loss_ema_d = EMA(torch.LongTensor(labels), num_classes=num_labels)


    num_epochs = 20
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    swap = False
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        if(epoch >= 1):
            swap = True
        train(model, train_dataloader, optimizer, device, swap, scheduler, sample_loss_ema_d, sample_loss_ema_b, args.dataset_name)
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
            if patience > 3:
                print("Early stopping at epoch : ",epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(best_output_model_path)
                patience = 0
                break
        model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_tokenizer_path)
    
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
