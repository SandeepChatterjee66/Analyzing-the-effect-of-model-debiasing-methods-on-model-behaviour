"""
This script trains a BERT-based model for  task using the Paraphrase identification using BERT.
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
import pickle
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset

from data_loader import load_paws, load_qqp

# from Generate_Embeddings.Paraphrase_identification.Feature_sieve.data_loader import load_paws, load_qqp


input_path = './'
output_path = 'resources'
log_soft = F.log_softmax
tokenizer_dir = "./tokenizer"
model_dir = "./model"
config_dir = "./config"
print(torch.version.cuda)
MAX_LEN = 512 # suitable for all datasets
MAX_GRAD_NORM = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
alpha1 = 0.2
alpha2 = 2
alpha3 = 0.9
num_labels = 2


class BiasModel(nn.Module):
    def __init__(self, attention_size, hidden_size, num_labels):
        """
        Initializes the BiasModel.

        Args:
            attention_size (int): The size of the attention layer.
            hidden_size (int): The size of the hidden layer.
            num_labels (int): The number of labels for classification.
        """
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size, attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)

    def forward(self, bert_hidden_layer_input):
        """
        Forward pass for the BiasModel.

        Args:
            bert_hidden_layer_input (Tensor): Input tensor from BERT model.

        Returns:
            Tensor: Logits from the classifier layer.
        """
        bert_hidden_layer_input = self.dropout(bert_hidden_layer_input)
        attention1_out = self.attention1(bert_hidden_layer_input)
        attention1_out = torch.tanh(attention1_out)
        attention2_out = self.attention2(attention1_out)
        attention2_out = F.softmax(attention2_out, dim=1)
        weighted_sum = torch.sum(attention2_out * bert_hidden_layer_input, dim=1)
        logits = self.classifier(weighted_sum)
        return logits


class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn=None):
        """
        Initializes the MainModel.

        Args:
            config (BertConfig): Configuration for the BERT model.
            loss_fn (callable, optional): Loss function for training. Defaults to None.
        """
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)
        self.bias_model = BiasModel(attention_size=1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2 * self.num_labels)
        self.classifier = nn.Linear(2 * self.num_labels, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, device):
        """
        Forward pass for the MainModel.

        Args:
            input_ids (Tensor): Input IDs.
            attention_mask (Tensor): Attention mask.
            token_type_ids (Tensor): Token type IDs.
            labels (Tensor): Ground truth labels.
            device (str): Device to run the model on.

        Returns:
            Tuple: Main loss, main predictions, bias loss, bias predictions, forget loss.
        """
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output.hidden_states[4]
        hidden_state_detached = hidden_state.detach()
        bias_prob = self.bias_model(hidden_state_detached)
        bias_loss = F.cross_entropy(bias_prob, labels.view(-1))
        forget_prob = self.bias_model(hidden_state)
        pseudo_labels = torch.ones_like(forget_prob) / self.num_labels
        forget_loss = F.cross_entropy(forget_prob, pseudo_labels)
        output = output.last_hidden_state
        output = output[:, 0, :]
        hidden_output = self.hidden_layer(output)
        main_prob = self.classifier(hidden_output)
        main_loss = F.cross_entropy(main_prob, labels.view(-1))
        main_pred = F.softmax(main_prob, dim=1)
        bias_pred = F.softmax(bias_prob, dim=1)
        return main_loss, main_pred, bias_loss, bias_pred, forget_loss

    def set_bias_grad(self, requires_grad):
        """
        Set requires_grad for bias model parameters.

        Args:
            requires_grad (bool): Whether gradients should be computed for bias model parameters.
        """
        for param in self.bias_model.parameters():
            param.requires_grad = requires_grad


def train(model, dataloader, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for training.
        device (str): Device to run the model on.
    """
    tr_loss, tr_accuracy = 0, 0
    total_bias_loss = 0
    total_forget_loss = 0
    tr_accuracy_bias = 0
    nb_tr_steps = 0
    model.train()

    for idx, batch in enumerate(dataloader):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        main_loss, main_pred, bias_loss, bias_pred, forget_loss = model(
            input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets, device=device
        )

        tr_loss += main_loss.item()
        total_bias_loss += bias_loss.item()
        total_forget_loss += forget_loss.item()
        nb_tr_steps += 1

        predicted_labels_main = torch.argmax(main_pred, dim=1)
        predicted_labels_bias = torch.argmax(bias_pred, dim=1)
        targets = targets.view(-1)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels_main.cpu().numpy())
        tmp_tr_accuracy_bias = accuracy_score(targets.cpu().numpy(), predicted_labels_bias.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        tr_accuracy_bias += tmp_tr_accuracy_bias

        if idx % 100 == 0:
            print(f'\tMain Model loss at {idx} steps: {tr_loss}')
            print(f'\tBias Model loss at {idx} steps: {total_bias_loss}')
            print(f'\tForget Model loss at {idx} steps: {total_forget_loss}')
            if idx != 0:
                print(f'\tMain Model Accuracy: {tr_accuracy / nb_tr_steps}')
                print(f'\tBias Model Accuracy: {tr_accuracy_bias / nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tMain Model Loss at {idx} steps: {tr_loss}\n')
                fh.write(f'\tBias Model Loss at {idx} steps: {total_bias_loss}\n')
                fh.write(f'\tForget Model Loss at {idx} steps: {total_forget_loss}\n')
                if idx != 0:
                    fh.write(f'\tMain Model Accuracy: {tr_accuracy / nb_tr_steps}')
                    fh.write(f'\tBias Model Accuracy: {tr_accuracy_bias / nb_tr_steps}')

        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
        optimizer.zero_grad()
        for param in model.bias_model.parameters():
            param.requires_grad = False
        torch.autograd.set_detect_anomaly(True)
        loss = alpha1 * main_loss
        if idx % 2 == 0:
            loss += alpha3 * forget_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        for param in model.bias_model.parameters():
            param.requires_grad = True
        optimizer.zero_grad()
        bias_loss = alpha2 * bias_loss
        bias_loss.backward()
        optimizer.step()

    print(f'\tMain Model loss for the epoch: {tr_loss}')
    print(f'\tBias Model loss for the epoch: {bias_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy / nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy: {tr_accuracy / nb_tr_steps}\n')


def read_dataset(data_path):
    """
    Read dataset from a pickle file.

    Args:
        data_path (str): Path to the pickle file.

    Returns:
        Any: Loaded data from the pickle file.
    """
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data


def valid(model, dataloader, device):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): DataLoader for validation data.
        device (str): Device to run the model on.

    Returns:
        Tuple: Validation loss and validation accuracy.
    """
    eval_loss = 0
    bias_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        loss_main, main_prob, loss_bias, _, _ = model(
            input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets, device=device
        )
        bias_loss += loss_bias.item()
        eval_loss += loss_main.item()
        nb_eval_steps += 1

        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy

    return eval_loss, eval_accuracy / nb_eval_steps


def main():
    """
    Main function to train and validate the model.
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
    best_output_model_path = output_model_path + '/BestModel'

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
    model = MainModel.from_pretrained("bert-base-uncased", config=config, loss_fn=None)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    data = read_dataset('../resources/QQP/train.pkl')
    eval_data = read_dataset('../resources/QQP/val.pkl')

    train_dataloader = DataLoader(data, shuffle=True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(eval_data, shuffle=True, batch_size=BATCH_SIZE)
    num_epochs = 20
    max_acc = 0.0
    patience = 0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch: {epoch + 1}\n')
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        print(f'\tValidation loss: {validation_loss}')
        print(f'\tValidation accuracy for epoch: {eval_acc}')
        with open('live.txt', 'a') as fh:
            fh.write(f'\tValidation Loss: {validation_loss}\n')
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
                print("Early stopping at epoch:", epoch)
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
        fh.write(f'Total training time: {total_time}\n')

    print(f"Total training time: {total_time}")


if __name__ == '__main__':
    main()