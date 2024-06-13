"""
This script trains a BERT-based model for Named Entity Recognition (NER) using the Huggingface library.
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

from data_loader import load_data
from util import EMA

# from Generate_Embeddings.Entity_classification.LfF.data_loader import load_data
# from Generate_Embeddings.Entity_classification.LfF.util import EMA

#to create an environment "conda create -n myenv python=3.9"
# Ignore all warnings
warnings.filterwarnings("ignore")


input_path = './'
output_path = 'resources'
log_soft = F.log_softmax
#tokenizer_dir = "./tokenizer"
#model_dir = "./model"
#config_dir = "./config"
print("torch.version.cuda",torch.version.cuda)
MAX_LEN = 512 # suitable for all datasets
MAX_GRAD_NORM = 10
#num_labels = 128
BATCH_SIZE = 36
LEARNING_RATE = 1e-5
#lambda_dis_align = 1
#lambda_swap_align = 1
#lambda_swap = 0.1
weight_decay = 0.001
step_size = 30
gamma = 0.6


class GeneralizedCELoss(nn.Module):
    """Generalized Cross Entropy Loss with custom weighting.

    This loss function modifies the gradient of the cross entropy loss based on the probability
    distribution of the logits and a parameter `q`. The loss weighting helps in handling noisy labels 
    in the dataset.

    Args:
        q (float): A parameter that controls the weighting of the loss. Default value is 0.7.
    """
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        """Computes the Generalized Cross Entropy Loss.

        Args:
            logits (torch.Tensor): The predicted logits from the model of shape (batch_size, num_classes).
            targets (torch.Tensor): The true labels of shape (batch_size).

        Returns:
            torch.Tensor: The computed loss.

        Raises:
            NameError: If the mean of the probability distribution or gathered probabilities is NaN.
        """
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        # modify gradient of cross entropy
        # loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        loss = (1-(Yg.squeeze()**self.q))/self.q
        

        # loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss

class MyModel(BertPreTrainedModel):
    """Custom BERT-based model for token classification.

    This model utilizes two instances of the BioBERT model, processes their outputs through
    linear layers, and combines the results for classification tasks.

    Args:
        config (transformers.PretrainedConfig): Configuration object with model parameters.
    """
    def __init__(self, config):
        super(MyModel,self).__init__(config)
        self.num_labels = config.num_labels
        print("********************************************************")
        print("number of labels",self.num_labels)
        self.model_l = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        self.model_b = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
        self.model_l.resize_token_embeddings(28998)
        self.model_b.resize_token_embeddings(28998) 
        
        self.fc_l_l = nn.Linear(768, 2*self.num_labels)
        self.o_l_l = nn.Linear(2*self.num_labels, self.num_labels)
        self.fc_l_b = nn.Linear(768, 2*self.num_labels)
        self.o_l_b = nn.Linear(2*self.num_labels, self.num_labels)

    def features(self,input_ids, attention_mask):
        """Extracts features from the input_ids using both instances of the BioBERT model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Last hidden states from both BioBERT models.
        """
        z_l = self.model_l(input_ids, attention_mask = attention_mask)
        z_b = self.model_b(input_ids, attention_mask = attention_mask)
        return z_l.last_hidden_state[:,0,:], z_b.last_hidden_state[:,0,:] #to extract representations of the first token (often the [CLS] token in BERT-based models) from the last hidden state.torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)hidden_size (int, optional, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.
    
    def linear(self, z_conflict, z_align):
        """Processes the extracted features through fully connected layers.

        Args:
            z_conflict (torch.Tensor): Concatenated hidden states for conflict resolution.
            z_align (torch.Tensor): Concatenated hidden states for alignment.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Outputs from the fully connected layers.
        """
        output1 = self.fc_l_l(z_conflict)
        output1 = self.o_l_l(output1)
        output2 = self.fc_l_b(z_align)
        output2 = self.o_l_b(output2)
        return output1,output2
 


def valid(model, dataloader, device):
    """Evaluates the model on the validation dataset.

    This function performs a forward pass through the model for each batch in the validation dataloader,
    calculates the loss and accuracy, and logs the results.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        device (torch.device): The device on which the model and data are loaded.

    Returns:
        Tuple[float, float]: Average loss and accuracy over the validation dataset.
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
            z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask)
            # z = torch.cat((z_l,z_b), dim = 1)
            pred, _ = model.linear(z_conflict = z_l, z_align = z_b)
            loss = F.cross_entropy(pred, targets.view(-1))
            eval_loss += loss
            pred = F.softmax(pred, dim = 1)
            predicted_labels = torch.argmax(pred, dim=1)
            targets = targets.view(-1)
            tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    print(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tValidation Loss for epoch: {eval_loss/nb_eval_steps}\n')
        fh.write(f'\tValidation accuracy for epoch: {eval_accuracy/nb_eval_steps}\n')
    return eval_loss/nb_eval_steps, eval_accuracy/nb_eval_steps 


def train(model, dataloader, optimizer, device,scheduler,sample_loss_ema_d,sample_loss_ema_b, dataset_name):
    """Trains the model on the given dataset.

    This function performs the training of the model by iterating over the dataloader, computing the loss,
    performing backpropagation, and updating the model parameters. It also calculates and logs the training loss
    and accuracy.

    Args:
        model (torch.nn.Module): The model to be trained.
        dataloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        device (torch.device): The device on which the model and data are loaded.
        swap (bool): Whether to perform the swapping operation during training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        sample_loss_ema_d (EMA): Exponential moving average for the loss of the conflict samples.
        sample_loss_ema_b (EMA): Exponential moving average for the loss of the align samples.
        dataset_name (str): The name of the dataset.

    Returns:
        None
    """
    tr_loss, total_tr_accuracy_main, total_tr_accuracy_main_swap = 0, 0, 0
    #tr_loss_bias = 0
    total_tr_accuracy_bias, total_tr_accuracy_bias_swap = 0,0
    bias_loss = 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    bias_criterion = GeneralizedCELoss(q = 0.7)
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        index = batch['index']
        #print("index_batch",batch["index"])
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)

        z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask)
        # z_conflict = torch.cat((z_l,z_b.detach()),dim = 1)
        # z_align = torch.cat((z_l.detach(), z_b),dim = 1)
        pred_conflict, pred_align = model.linear(z_conflict = z_l, z_align = z_b)
        loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none').detach()
        loss_dis_align = F.cross_entropy(pred_align,targets.view(-1), reduction = 'none').detach()
        #tr_loss_bias+= loss_dis_align
        # EMA sample loss
        sample_loss_ema_d.update(loss_dis_conflict, index)
        sample_loss_ema_b.update(loss_dis_align, index)

        # class-wise normalize
        loss_dis_conflict = sample_loss_ema_d.parameter[index].clone().detach()
        loss_dis_align = sample_loss_ema_b.parameter[index].clone().detach()

        loss_dis_conflict = loss_dis_conflict.to(device)
        loss_dis_align = loss_dis_align.to(device)

        for c in range(num_labels):
            class_index = torch.where(targets == c)[0].to(device)
            max_loss_conflict = sample_loss_ema_d.max_loss(c)
            max_loss_align = sample_loss_ema_b.max_loss(c)
            loss_dis_conflict[class_index] /= max_loss_conflict
            loss_dis_align[class_index] /= max_loss_align

        loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
        # print(f"loss weights : {loss_weight[0]}")

        
        loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction = 'none')
        tr_loss += loss_dis_conflict.mean().item()
        
        loss_dis_conflict = loss_dis_conflict * loss_weight.to(device)
        # print(f"New loss : {loss_dis_conflict[0]}")
        loss_dis_align = bias_criterion(pred_align, targets.view(-1))
        # if(idx % 100 == 0):
        #     print(f'\tStep {idx} : ')
        #     print(f"\t\tLoss_Dis conflict : {loss_dis_conflict.mean()}")
        #     print(f"\t\tLoss_Dis swap : {loss_dis_align.mean()}")
        loss_dis = loss_dis_conflict.mean() + loss_dis_align.mean()
        loss = loss_dis
        # print(f'\tLoss Main : {loss_main}')
        
        nb_tr_steps += 1
        # with open('loss.csv', 'a', newline = '') as fh:
        #     if(flag == 0):
        #         fh.write(f'{(loss_dis_conflict.mean()).item()}, {(loss_dis_align.mean()).item()}, 0.0, 0.0\n')
        #     else:
        #         fh.write(f'{(loss_dis_conflict.mean()).item()}, {(loss_dis_align.mean()).item()}, {(loss_swap_conflict.mean()).item()}, {(loss_swap_align.mean()).item()}\n')
        #compute training accuracy
        pred_conflict = F.softmax(pred_conflict, dim = 1)
        pred_align = F.softmax(pred_align, dim = 1)
        # if flag == 1:
        #     pred_swap_conflict = F.softmax(pred_swap_conflict, dim = 1)
        #     pred_swap_align = F.softmax(pred_swap_align, dim = 1)

        targets = targets.view(-1)
        # print(targets.shape)
        predicted_labels_conflict = torch.argmax(pred_conflict,dim=1)
        # print(predicted_labels.shape)
        tr_accuracy_main = accuracy_score(targets.cpu().numpy(), predicted_labels_conflict.cpu().numpy())

        predicted_labels_align = torch.argmax(pred_align,dim=1)
        # print(predicted_labels.shape)
        tr_accuracy_bias = accuracy_score(targets.cpu().numpy(), predicted_labels_align.cpu().numpy())

        # if(flag == 1):
        #     predicted_labels_conflict = torch.argmax(pred_swap_conflict,dim=1)
        #     tr_accuracy_main_swap = accuracy_score(targets.cpu().numpy(), predicted_labels_conflict.cpu().numpy())
        #     predicted_labels_align = torch.argmax(pred_swap_align,dim=1)
        #     tr_accuracy_bias_swap = accuracy_score(targets.cpu().numpy(), predicted_labels_align.cpu().numpy())
            
        
        with open('accuracy.csv','a', newline ='') as fh:
             fh.write(f'{tr_accuracy_main}, {tr_accuracy_bias}\n')
           # if(flag == 0):
           #    fh.write(f'{tr_accuracy_main}, 0.0, {tr_accuracy_bias}, 0.0\n')
           #else:
           #   fh.write(f'{tr_accuracy_main}, {tr_accuracy_main_swap}, {tr_accuracy_bias}, {tr_accuracy_bias_swap}\n')
        total_tr_accuracy_main += tr_accuracy_main
        total_tr_accuracy_bias += tr_accuracy_bias
        # if(flag == 1):
        #     total_tr_accuracy_main_swap += tr_accuracy_main_swap
        #     total_tr_accuracy_bias_swap += tr_accuracy_bias_swap
        if idx % 100 == 0:
            #print(f'\t\tBias model loss: {tr_loss_bias/nb_tr_steps}')
            print(f'\t\tMain model loss: {tr_loss/nb_tr_steps}')
            print(f'\t\tMain Model Accuracy : {total_tr_accuracy_main/nb_tr_steps}')
           # print(f'\t\tSwap Main Model Accuracy : {total_tr_accuracy_main_swap/nb_tr_steps}')
            print(f'\t\tBias model Accuracy : {total_tr_accuracy_bias/nb_tr_steps}')
           # print(f'\t\tSwap Bias model Accuracy : {total_tr_accuracy_bias_swap/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                #fh.write(f'\tBias Model Loss at {idx} steps : {tr_loss_bias/nb_tr_steps}\n')
                fh.write(f'\tMain Model Loss at {idx} steps : {tr_loss/nb_tr_steps}\n')
                fh.write(f'\tMain Model Accuracy : {total_tr_accuracy_main/nb_tr_steps}')
               # fh.write(f'\tSwap Main Model Accuracy : {total_tr_accuracy_main_swap/nb_tr_steps}')
                fh.write(f'\tBias model Accuracy : {total_tr_accuracy_bias/nb_tr_steps}')
               # fh.write(f'\tSwap Bias model Accuracy : {total_tr_accuracy_bias_swap/nb_tr_steps}')

        # a technique to prevent the exploding gradient problem during training in deep neural networks
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(),
            max_norm = 10
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if swap == True:
            #scheduler.step()
       



    print(f'\tMain model loss for the epoch: {tr_loss/nb_tr_steps}')
    print(f'\tTraining accuracy of main model for epoch: {total_tr_accuracy_main/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy of main model for epoch: {total_tr_accuracy_main/nb_tr_steps}\n')
        fh.write(f'\tTraining Loss of main model for epoch: {tr_loss/nb_tr_steps}\n')


def main():
    """Main function to train a model on a specified dataset.

    This function performs the following steps:
    1. Parse command line arguments to get dataset name, model output directory, and tokenizer output directory.
    2. Prepare output directories for model and tokenizer.
    3. Log the dataset name and output paths to a file.
    4. Determine the number of labels based on the dataset name.
    5. Load the tokenizer and data for training and validation.
    6. Initialize the model, optimizer, scheduler, and other necessary components.
    7. Train the model for a specified number of epochs with early stopping.
    8. Save the trained model and tokenizer to the specified directories.
    9. Log the total training time to a file and print it.

    Args:
        --dataset_name (str): Name of the dataset to be used for training.
        --output_model_directory (str): Directory to save the trained model.
        --output_tokenizer_directory (str): Directory to save the tokenizer.

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
    print("dataset_name",args.dataset_name)
    global num_labels
    if args.dataset_name == "BC5CDR":
        num_labels = 2
    elif args.dataset_name == "MedMentions":
        num_labels = 128
    print("num_labels :",num_labels)
    print(f'Dataset : {args.dataset_name}')
    print(f'Number of labels : {num_labels}')
    
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

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
    
    # special_tokens_dict = {'additional_special_tokens':['[ENTITY_START]','[ENTITY_END]']}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'tokens')
    
    id2label, label2id, train_data, labels = load_data(args.dataset_name, 'train', tokenizer)
    # print(train_data[0])
    # print(len(label2id))
    _,_,devel_data,_ = load_data(args.dataset_name, 'devel', tokenizer)
    config = AutoConfig.from_pretrained("dmis-lab/biobert-v1.1" , num_labels=num_labels)
    model = MyModel.from_pretrained("dmis-lab/biobert-v1.1", config=config)
    #model = MyModel.from_pretrained("dmis-lab/biobert-v1.1")
    #model.model_l.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    #model.model_b.resize_token_embeddings(len(tokenizer)) 
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)

    labels = [label2id[label] for label in labels]

    sample_loss_ema_b = EMA(torch.LongTensor(labels), num_classes=num_labels)
    sample_loss_ema_d = EMA(torch.LongTensor(labels), num_classes=num_labels)
        

    train_dataloader = DataLoader(train_data, shuffle = True, batch_size=BATCH_SIZE)
    devel_dataloader = DataLoader(devel_data, shuffle=True, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #Decays the learning rate of each parameter group by gamma every step_size epochs.

    num_epochs = 35
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    # swap = False
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        #if(epoch >= 1):
         #   swap = True
        train(model, train_dataloader, optimizer, device, scheduler, sample_loss_ema_d, sample_loss_ema_b, args.dataset_name)
        validation_loss, eval_acc = valid(model, devel_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')
        
        if eval_acc > max_acc:
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
     

