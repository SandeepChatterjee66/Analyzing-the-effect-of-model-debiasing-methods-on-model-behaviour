"""
This script is designed to train a model on a dataset and save the best model and tokenizer.

Usage::

    python train_model.py --dataset_name <dataset_name> --output_model_directory <output_model_directory> --output_tokenizer_directory <output_tokenizer_directory>

Arguments::

    --dataset_name (str): Name of the dataset.
    --output_model_directory (str): Path to the directory to save the trained model.
    --output_tokenizer_directory (str): Path to the directory to save the tokenizer.

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
from Generate_Embeddings.Paraphrase_identification.LfF.data_loader import load_qqp, load_paws
from Generate_Embeddings.Paraphrase_identification.LfF.test import inference
from Generate_Embeddings.Paraphrase_identification.LfF.util import EMA
import numpy as np

input_path = './'
log_soft = F.log_softmax
#tokenizer_dir = "./tokenizer"
#model_dir = "./model"
#config_dir = "./config"
print("torch.version.cuda",torch.version.cuda)
MAX_LEN = 512 # suitable for all datasets
MAX_GRAD_NORM = 10

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
#lambda_dis_align = 1
#lambda_swap_align = 1
#lambda_swap = 0.1
weight_decay = 0.001
step_size = 30
gamma = 0.6
num_labels = 2

class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss.

    Args:
        q (float): Parameter for controlling the hardness of the classification task.
                   Default is 0.7.
    """

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        """
        Forward pass of the loss.

        Args:
            logits (torch.Tensor): The predicted logits.
            targets (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: Computed loss.
        """
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')
        loss = (1 - (Yg.squeeze() ** self.q)) / self.q

        return loss


class MainModel(BertPreTrainedModel):
    """
    Main Model class.

    Args:
        config (AutoConfig): Configuration for the model.
    """

    def __init__(self, config):
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        print("********************************************************")
        print("number of labels", self.num_labels)
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")

        self.fc_l_l = nn.Linear(768, 2 * self.num_labels)
        self.o_l_l = nn.Linear(2 * self.num_labels, self.num_labels)
        self.fc_l_b = nn.Linear(768, 2 * self.num_labels)
        self.o_l_b = nn.Linear(2 * self.num_labels, self.num_labels)

    def features(self, input_ids, attention_mask, token_type_ids):
        """
        Extract features from the input.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            token_type_ids (torch.Tensor): Token type IDs.

        Returns:
            tuple: Tuple containing the features.
        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        z_b = self.model_b(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return z_l.last_hidden_state[:, 0, :], z_b.last_hidden_state[:, 0, :]  # to extract representations of the first token (often the [CLS] token in BERT-based models) from the last hidden state.torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)hidden_size (int, optional, defaults to 768) — Dimensionality of the encoder layers and the pooler layer.

    def linear(self, z_conflict, z_align):
        """
        Perform linear transformation.

        Args:
            z_conflict (torch.Tensor): Features for conflict model.
            z_align (torch.Tensor): Features for alignment model.

        Returns:
            tuple: Tuple containing the outputs.
        """
        output1 = self.fc_l_l(z_conflict)
        output1 = self.o_l_l(output1)
        output2 = self.fc_l_b(z_align)
        output2 = self.o_l_b(output2)
        return output1, output2


def valid(model, dataloader, device):
    """
    Validate the model.

    Args:
        model (MainModel): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run the validation on.

    Returns:
        tuple: Tuple containing validation loss and accuracy.
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
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        nb_eval_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask, token_type_ids=token_type_ids)
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
    """Train the model.

    Args:
        model (MainModel): The main model to be trained.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        device (str): Device to run the training on (e.g., 'cpu', 'cuda').
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        sample_loss_ema_d (EMA): Exponential Moving Average for sample loss (conflict).
        sample_loss_ema_b (EMA): Exponential Moving Average for sample loss (bias).
        dataset_name (str): Name of the dataset being used.

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
        #index = batch['index']
        #print("index_batch",batch["index"])
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask, token_type_ids=token_type_ids)
        # z_conflict = torch.cat((z_l,z_b.detach()),dim = 1)
        # z_align = torch.cat((z_l.detach(), z_b),dim = 1)
        pred_conflict, pred_align = model.linear(z_conflict = z_l, z_align = z_b)
        loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none').detach()
        loss_dis_align = F.cross_entropy(pred_align,targets.view(-1), reduction = 'none').detach()
        #tr_loss_bias+= loss_dis_align
        # EMA sample loss
        # sample_loss_ema_d.update(loss_dis_conflict, index)
        # sample_loss_ema_b.update(loss_dis_align, index)
        sample_loss_ema_d.update(loss_dis_conflict, idx)
        sample_loss_ema_b.update(loss_dis_align, idx)



        # class-wise normalize
        # loss_dis_conflict = sample_loss_ema_d.parameter[index].clone().detach()
        # loss_dis_align = sample_loss_ema_b.parameter[index].clone().detach()
        loss_dis_conflict = sample_loss_ema_d.parameter[idx].clone().detach()
        loss_dis_align = sample_loss_ema_b.parameter[idx].clone().detach()

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
        loss_dis = loss_dis_conflict.mean() + loss_dis_align.mean()
        loss = loss_dis
        # print(f'\tLoss Main : {loss_main}')
        
        nb_tr_steps += 1
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
            print(f'\tStep {idx}:')
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
    """Main function for training the model."""
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

    # train_file_path = os.path.join(input_path, args.dataset_name, args.dataset_name + '_train.txt')
    # dev_matched_file_path = os.path.join(input_path, args.dataset_name, args.dataset_name + '_dev_matched.txt')
    # dev_mismatched_file_path = os.path.join(input_path, args.dataset_name, args.dataset_name + '_dev_mismatched.txt')

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ## data, label2id, labels = load_mnli(file_path=train_file_path, tokenizer=tokenizer)
    ## print(data[0])

    config = AutoConfig.from_pretrained("bert-base-uncased" , num_labels=num_labels)
    model = MainModel.from_pretrained("bert-base-uncased", config = config)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma) #Decays the learning rate of each parameter group by gamma every step_size epochs.

    #dev_data1,_,_ = load_mnli(file_path=dev_matched_file_path, tokenizer=tokenizer, type = False)
    #dev_data2,_,_ = load_mnli(file_path=dev_mismatched_file_path, tokenizer=tokenizer,type = False)
    #eval_data = ConcatDataset([dev_data1, dev_data2])

    data = read_dataset('./QQP/train.pkl')
    eval_data = read_dataset('./QQP/val.pkl')

    lables_file = './QQP/labels_clean.txt'
    with open(lables_file, 'r') as infile:
        labels = [int(line.strip()) for line in infile]
        print(f'labels length {len(labels)}')

    #labels = [label2id[label] for label in labels]

    sample_loss_ema_b = EMA(torch.LongTensor(labels), num_classes=num_labels)
    sample_loss_ema_d = EMA(torch.LongTensor(labels), num_classes=num_labels)
    # mnli_mm_dataloader = DataLoader(dev_data2, shuffle = True, batch_size=BATCH_SIZE)
    # hans_data = load_hans(file_path='./HANS/hans1.txt', tokenizer=tokenizer)
    # hans_dataloader = DataLoader(hans_data, shuffle = True, batch_size=BATCH_SIZE)


    train_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=BATCH_SIZE)
    #paws_test_dataloader = DataLoader(paws_data, shuffle=False)
    
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
        #paws_acc = inference(model, paws_test_dataloader, tokenizer, device)

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