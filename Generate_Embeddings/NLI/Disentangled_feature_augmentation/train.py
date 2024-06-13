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
import gc
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
import warnings
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

from data_loader import load_mnli
from data_loader import load_hans
from data_loader import load_snli
from test import inference
from test import read_dataset

from Generate_Embeddings.NLI.Disentangled_feature_augmentation.util import EMA


# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_mnli
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_hans
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import load_snli
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.test import inference
# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.test import read_dataset

# from Generate_Embeddings.NLI.Disentangled_feature_augmentation.util import EMA

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
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
lambda_dis_align = 1
lambda_swap_align = 0
lambda_swap = 0.1
weight_decay = 0.001
step_size = 30
gamma = 0.6

class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss function.

    This loss function is an extension of Cross Entropy Loss (CE) with a
    parameter `q` which allows to control the focus of the loss function
    towards hard or easy examples.

    Args:
        q (float): A parameter controlling the focus of the loss function.
            The default value is 0.7.
    """

    def __init__(self, q=0.7):
        """
        Initializes the GeneralizedCELoss module.

        Args:
            q (float): A parameter controlling the focus of the loss function.
                The default value is 0.7.
        """
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        """
        Computes the Generalized Cross Entropy loss.

        Args:
            logits (torch.Tensor): Raw predictions from the model.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: The computed loss.

        Raises:
            NameError: If the computed probability or Yg (gathered probabilities)
                contains NaN values.

        Note:
            This implementation assumes a single-label classification task.

        """
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class MainModel(BertPreTrainedModel):
    """
    Main model architecture for dual-BERT model.

    This model architecture consists of two BERT models, each followed by
    linear layers for conflict and alignment tasks.

    Args:
        config (BertConfig): Configuration for the BERT model.
    """

    def __init__(self, config):
        """
        Initializes the MainModel module.

        Args:
            config (BertConfig): Configuration for the BERT model.
        """
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_l_2 = nn.Linear(2*(self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_b_2 = nn.Linear(2*(self.num_labels), self.num_labels)

    def features(self,input_ids, attention_mask):
        """
        Extracts features from the BERT models.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for input.

        Returns:
            torch.Tensor: Features for conflict task.
            torch.Tensor: Features for alignment task.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask)
        z_b = self.model_b(input_ids, attention_mask=attention_mask)
        return z_l.last_hidden_state[:,0,:], z_b.last_hidden_state[:,0,:]
    
    def linear(self, z_conflict, z_align):
        """
        Performs linear transformations on the features.

        Args:
            z_conflict (torch.Tensor): Features for conflict task.
            z_align (torch.Tensor): Features for alignment task.

        Returns:
            torch.Tensor: Output for conflict task.
            torch.Tensor: Output for alignment task.
        """
        hidden_output1 = self.fc_l_1(z_conflict)
        output1 = self.fc_l_2(hidden_output1)
        hidden_output2 = self.fc_b_1(z_align)
        output2 = self.fc_b_2(hidden_output2)
        return output1,output2

 


def valid(model, dataloader, device):
    """
    Performs validation on the provided data loader.

    Args:
        model (MainModel): The model to evaluate.
        dataloader (DataLoader): DataLoader containing the validation data.
        device (torch.device): Device to perform computation on.

    Returns:
        tuple: A tuple containing validation loss and accuracy.

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
    return eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps



def train(model, dataloader, optimizer, device, swap, scheduler, sample_loss_ema_d, sample_loss_ema_b, dataset_name):
    """
    Trains the provided model on the given data.

    Args:
        model (MainModel): The model to train.
        dataloader (DataLoader): DataLoader containing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (torch.device): Device to perform computation on.
        swap (bool): Flag indicating whether to perform swapping.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        sample_loss_ema_d (EMA): Exponential moving average for loss in conflict task.
        sample_loss_ema_b (EMA): Exponential moving average for loss in alignment task.
        dataset_name (str): Name of the dataset.

    Returns:
        None
    """
    tr_loss, total_tr_accuracy_main, total_tr_accuracy_main_swap = 0, 0, 0
    total_tr_accuracy_bias, total_tr_accuracy_bias_swap = 0, 0
    bias_loss = 0 # bias loss
    nb_tr_steps = 0
    bias_criterion = GeneralizedCELoss(q=0.7)
    model.train()
    
    for idx, batch in enumerate(dataloader):
        index = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask)
        z_conflict = torch.cat((z_l, z_b.detach()), dim=1)
        z_align = torch.cat((z_l.detach(), z_b), dim=1)
        pred_conflict, pred_align = model.linear(z_conflict=z_conflict, z_align=z_align)
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
        loss_dis_conflict = loss_dis_conflict * loss_weight.to(device)
        loss_dis_align = bias_criterion(pred_align, targets.view(-1))

        loss_dis = loss_dis_conflict.mean() + lambda_dis_align * loss_dis_align.mean()
        flag = 0

        if dataset_name == 'BC5CDR' and idx > 60:
            swap = True
        if swap == True:
            flag = 1
            indices = np.random.permutation(z_b.size(0))
            z_b_swap = z_b[indices]
            targets_swap = targets[indices]
            z_swap_conflict = torch.cat((z_l, z_b_swap.detach()), dim=1)
            z_swap_align = torch.cat((z_l.detach(), z_b_swap), dim=1)
            pred_swap_conflict, pred_swap_align = model.linear(z_conflict=z_swap_conflict, z_align=z_swap_align)
            loss_swap_conflict = F.cross_entropy(pred_swap_conflict, targets.view(-1), reduction='none')
            loss_swap_conflict = loss_swap_conflict * loss_weight.to(device)
            loss_swap_align = bias_criterion(pred_swap_align, targets_swap.view(-1))
            loss_swap = loss_swap_conflict.mean() + lambda_swap_align * loss_swap_align.mean()

            loss = loss_dis + lambda_swap * loss_swap
        else:
            loss = loss_dis

        tr_loss += loss.item()
        nb_tr_steps += 1

        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=10
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'\tModel loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {total_tr_accuracy_main/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {total_tr_accuracy_main/nb_tr_steps}\n')


def main():
    """
    Main function to train the model.

    Returns:
        None
    """
    gc.collect()
    torch.cuda.empty_cache()
    print("Training model:")
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)

    args = parser.parse_args()

    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory)

    best_hans_model_path = os.path.join(output_model_path, 'BestHANSModel')   
    best_mnli_mm_model_path = os.path.join(output_model_path, 'BestMNLI_mmModel')   
    best_output_model_path = os.path.join(output_model_path, 'BestModel')    

    for path in [output_model_path, best_output_model_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    train_file_path = os.path.join('../resources', args.dataset_name, args.dataset_name + '_train.txt')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data, label2id, labels = load_mnli(file_path=train_file_path, tokenizer=tokenizer)

    config = AutoConfig.from_pretrained("bert-base-uncased" , num_labels=num_labels)
    model = MainModel.from_pretrained("bert-base-uncased", config=config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) 

    eval_data = read_dataset('../resources/multinli_1.0/val.pkl')

    labels = [label2id[label] for label in labels]

    sample_loss_ema_b = EMA(torch.LongTensor(labels), num_classes=num_labels)
    sample_loss_ema_d = EMA(torch.LongTensor(labels), num_classes=num_labels)

    hans_data = load_hans(file_path='../resources/HANS/hans1.txt', tokenizer=tokenizer)
    hans_dataloader = DataLoader(hans_data, shuffle=True, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=BATCH_SIZE)
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

