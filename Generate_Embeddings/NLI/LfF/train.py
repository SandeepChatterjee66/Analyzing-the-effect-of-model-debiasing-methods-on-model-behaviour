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
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from Generate_Embeddings.NLI.LfF.data_loader import load_mnli
from Generate_Embeddings.NLI.LfF.data_loader import load_hans
from Generate_Embeddings.NLI.LfF.data_loader import load_snli
from Generate_Embeddings.NLI.LfF.test import inference
from Generate_Embeddings.NLI.LfF.test import read_dataset
from Generate_Embeddings.NLI.LfF.util import EMA
import numpy as np


input_path = './'
output_path = 'resources'
log_soft = F.log_softmax
print("torch.version.cuda",torch.version.cuda)
MAX_LEN = 512 # suitable for all datasets
MAX_GRAD_NORM = 10

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
weight_decay = 0.001
step_size = 30
gamma = 0.6
num_labels = 3

class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss.

    Args:
        q (float, optional): Exponent parameter. Defaults to 0.7.
    """

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        """
        Forward pass of the loss function.

        Args:
            logits (torch.Tensor): The predicted logits.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed loss value.
        """
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = (1-(Yg.squeeze()**self.q))/self.q
        
        return loss

class MainModel(BertPreTrainedModel):
    """
    Main Model class for NLI task.

    Args:
        config (AutoConfig): The configuration object.
    """

    def __init__(self, config):
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        print("********************************************************")
        print("number of labels",self.num_labels)
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        
        self.o_l_l = nn.Linear(768, self.num_labels)
        self.o_l_b = nn.Linear(768, self.num_labels)

    def features(self,input_ids, attention_mask, token_type_ids):
        """
        Extracts features from the input.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attention_mask (torch.Tensor): The attention mask.
            token_type_ids (torch.Tensor): The token type IDs.

        Returns:
            torch.Tensor: The features extracted.
        """
        z_l = self.model_l(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        z_b = self.model_b(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        return z_l.last_hidden_state[:,0,:], z_b.last_hidden_state[:,0,:] 
        # to extract representations of the first token (often the [CLS] token in BERT-based models) from
        # the last hidden state.torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # hidden_size (int, optional, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
    
    def linear(self, z_conflict, z_align):
        """
        Performs linear transformation.

        Args:
            z_conflict (torch.Tensor): Features for the conflict.
            z_align (torch.Tensor): Features for the alignment.

        Returns:
            torch.Tensor: The linear transformed outputs.
        """
        output1 = self.o_l_l(z_conflict)
        output2 = self.o_l_b(z_align)
        return output1,output2

    
        
def valid(model, dataloader, device):
    """
    Performs validation on the model.

    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (str): Device to run the validation on.

    Returns:
        tuple: A tuple containing the validation loss and accuracy.
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
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        nb_eval_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
            pred, _ = model.linear(z_conflict=z_l, z_align=z_b)
            loss = F.cross_entropy(pred, targets.view(-1))
            eval_loss += loss
            pred = F.softmax(pred, dim=1)
            predicted_labels = torch.argmax(pred, dim=1)
            targets = targets.view(-1)
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss for epoch: {eval_loss/nb_eval_steps}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps 



def train(model, dataloader, optimizer, device, scheduler, sample_loss_ema_d, sample_loss_ema_b, dataset_name):
    """
    Trains the model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to run the training on.
        scheduler: Learning rate scheduler.
        sample_loss_ema_d: Exponential Moving Average (EMA) for sample loss (conflict).
        sample_loss_ema_b: Exponential Moving Average (EMA) for sample loss (align).
        dataset_name (str): Name of the dataset.

    Returns:
        None
    """
    tr_loss, total_tr_accuracy_main, total_tr_accuracy_main_swap = 0, 0, 0
    total_tr_accuracy_bias, total_tr_accuracy_bias_swap = 0, 0
    bias_loss = 0
    nb_tr_steps = 0
    bias_criterion = GeneralizedCELoss(q=0.7)
    model.train()

    for idx, batch in enumerate(dataloader):
        index = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids)
        pred_conflict, pred_align = model.linear(z_conflict=z_l, z_align=z_b)
        loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none').detach()
        loss_dis_align = F.cross_entropy(pred_align, targets.view(-1), reduction='none').detach()

        sample_loss_ema_d.update(loss_dis_conflict, index)
        sample_loss_ema_b.update(loss_dis_align, index)

        loss_dis_conflict = sample_loss_ema_d.parameter[index].clone().detach().to(device)
        loss_dis_align = sample_loss_ema_b.parameter[index].clone().detach().to(device)

        for c in range(num_labels):
            class_index = torch.where(targets == c)[0].to(device)
            max_loss_conflict = sample_loss_ema_d.max_loss(c)
            max_loss_align = sample_loss_ema_b.max_loss(c)
            loss_dis_conflict[class_index] /= max_loss_conflict
            loss_dis_align[class_index] /= max_loss_align

        loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
        
        loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
        tr_loss += loss_dis_conflict.mean().item()
        
        loss_dis_conflict = loss_dis_conflict * loss_weight
        loss_dis_align = bias_criterion(pred_align, targets.view(-1))
        loss_dis = loss_dis_conflict.mean() + loss_dis_align.mean()

        loss = loss_dis

        nb_tr_steps += 1
        pred_conflict = F.softmax(pred_conflict, dim=1)
        pred_align = F.softmax(pred_align, dim=1)
 
        targets = targets.view(-1)
        predicted_labels_conflict = torch.argmax(pred_conflict, dim=1)
        tr_accuracy_main = accuracy_score(targets.cpu().numpy(), predicted_labels_conflict.cpu().numpy())

        predicted_labels_align = torch.argmax(pred_align, dim=1)
        tr_accuracy_bias = accuracy_score(targets.cpu().numpy(), predicted_labels_align.cpu().numpy())
            
        with open('accuracy.csv','a', newline ='') as fh:
            fh.write(f'{tr_accuracy_main}, {tr_accuracy_bias}\n')
        total_tr_accuracy_main += tr_accuracy_main
        total_tr_accuracy_bias += tr_accuracy_bias

        if idx % 100 == 0:
            print(f'\tStep {idx}:')
            print(f'\t\tMain model loss: {tr_loss/nb_tr_steps}')
            print(f'\t\tMain Model Accuracy : {total_tr_accuracy_main/nb_tr_steps}')
            print(f'\t\tBias model Accuracy : {total_tr_accuracy_bias/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tMain Model Loss at {idx} steps : {tr_loss/nb_tr_steps}\n')
                fh.write(f'\tMain Model Accuracy : {total_tr_accuracy_main/nb_tr_steps}')
                fh.write(f'\tBias model Accuracy : {total_tr_accuracy_bias/nb_tr_steps}')

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=10
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step with the scheduler after backward pass.

    print(f'\tMain model loss for the epoch: {tr_loss/nb_tr_steps}')
    print(f'\tTraining accuracy of main model for epoch: {total_tr_accuracy_main/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy of main model for epoch: {total_tr_accuracy_main/nb_tr_steps}\n')
        fh.write(f'\tTraining Loss of main model for epoch: {tr_loss/nb_tr_steps}\n')

    
def main():
    """
    Main function to execute the training process.
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

    train_file_path = os.path.join('../resources', args.dataset_name, args.dataset_name + '_train.txt')

    tokenizer = AutoTokenizer.from_pretrained(best_output_model_path)
    data, label2id, labels = load_mnli(file_path=train_file_path, tokenizer=tokenizer)

    config = AutoConfig.from_pretrained(best_output_model_path, num_labels=num_labels)
    model = MainModel.from_pretrained(best_output_model_path, config=config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    eval_data = read_dataset('..resources/multinli_1.0/val.pkl')

    labels = [label2id[label] for label in labels]

    sample_loss_ema_b = EMA(torch.LongTensor(labels), num_classes=num_labels)
    sample_loss_ema_d = EMA(torch.LongTensor(labels), num_classes=num_labels)
    hans_data = load_hans(file_path='./HANS/hans1.txt', tokenizer=tokenizer)
    hans_dataloader = DataLoader(hans_data, shuffle=True, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=BATCH_SIZE)
    num_epochs = 20
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device, scheduler, sample_loss_ema_d, sample_loss_ema_b, args.dataset_name)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        hans_acc, _, _ = inference(model, hans_dataloader, tokenizer, device, data='hans')
        print(f'\tValidation loss: {validation_loss}')
        print(f'\tValidation accuracy for epoch: {eval_acc}')
        print(f'\tHANS accuracy: {hans_acc}')
        with open('live.txt', 'a') as fh:
            fh.write(f'\tValidation Loss : {validation_loss}\n')
            fh.write(f'\tValidation accuracy for epoch: {eval_acc}\n')
            fh.write(f'\tHANS accuracy: {hans_acc}\n')
        
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
                print("Early stopping at epoch : ", epoch)
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

