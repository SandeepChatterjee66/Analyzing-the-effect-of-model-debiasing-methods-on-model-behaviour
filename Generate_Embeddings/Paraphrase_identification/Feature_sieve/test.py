"""
This script test the model trained for Natural Language Inference(NLI) task. It saves the prediction file and the softmax score file

Usage::

    python test.py --input_model_path <path_to_pretrained_model> --paws_test_path <path_to_paws_test_file>

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    --paws_test_path (str) : Path to the paws test dataset
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
from Generate_Embeddings.Paraphrase_identification.Feature_sieve.data_loader import load_qqp
from Generate_Embeddings.Paraphrase_identification.Feature_sieve.data_loader import load_paws

input_path = './'
input_model_path = './best_model'
BATCH_SIZE = 32


class BiasModel(nn.Module):
    """Bias model to predict bias in the input data using attention mechanism.

    Args:
        attention_size (int): Size of the attention layer.
        hidden_size (int): Size of the hidden layer.
        num_labels (int): Number of output labels.

    Attributes:
        num_labels (int): Number of output labels.
        dropout (nn.Dropout): Dropout layer.
        attention1 (nn.Linear): First attention layer.
        attention2 (nn.Linear): Second attention layer.
        classifier (nn.Linear): Classification layer.
    """
    def __init__(self, attention_size, hidden_size, num_labels):
        super(BiasModel, self).__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.1)  
        self.attention1 = nn.Linear(hidden_size, hidden_size)
        self.attention2 = nn.Linear(hidden_size, attention_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        
    def forward(self, bert_hidden_layer_input):
        """Forward pass through the BiasModel.

        Args:
            bert_hidden_layer_input (torch.Tensor): Input tensor from the BERT model.

        Returns:
            torch.Tensor: Logits for the classification.
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
    """Main model incorporating a bias model and a main classifier.

    Args:
        config (transformers.PretrainedConfig): Configuration for the BERT model.
        loss_fn (callable, optional): Loss function for the model.

    Attributes:
        num_labels (int): Number of output labels.
        loss_fn (callable): Loss function.
        bert (transformers.AutoModel): BERT model.
        bias_model (BiasModel): Bias model.
        hidden_layer (nn.Linear): Hidden layer.
        classifier (nn.Linear): Classification layer.
    """
    def __init__(self, config, loss_fn=None):
        super(MainModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("bert-base-uncased", config=config)
        self.bias_model = BiasModel(attention_size=1, hidden_size=768, num_labels=self.num_labels)
        self.hidden_layer = nn.Linear(768, 2 * self.num_labels)
        self.classifier = nn.Linear(2 * self.num_labels, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, device):
        """Forward pass through the MainModel.

        Args:
            input_ids (torch.Tensor): Input IDs for the BERT model.
            attention_mask (torch.Tensor): Attention mask for the BERT model.
            token_type_ids (torch.Tensor): Token type IDs for the BERT model.
            labels (torch.Tensor): True labels for the input data.
            device (torch.device): Device to run the model on.

        Returns:
            tuple: Main loss, main predictions, bias loss, bias predictions, and forget loss.
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
        """Set the requires_grad attribute for all parameters in the bias model.

        Args:
            requires_grad (bool): If True, gradients will be calculated for the bias model parameters.
        """
        for param in self.bias_model.parameters():
            param.requires_grad = requires_grad


def inference(model, dataloader, tokenizer, device, data='mnli'):
    """Run inference on the given model and dataloader.

    Args:
        model (nn.Module): The model to run inference on.
        dataloader (torch.utils.data.DataLoader): DataLoader for the inference data.
        tokenizer (transformers.AutoTokenizer): Tokenizer used for the input data.
        device (torch.device): Device to run the model on.
        data (str): Type of data, either 'mnli' or 'hans'.

    Returns:
        tuple: Test accuracy, predictions list, and probability list.
    """
    model.eval()
    pred_lst = []
    prob_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    hans_dict = {1: 'non-entailment', 2: 'entailment'}
    mnli_dict = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        with torch.no_grad():
            loss_main, main_prob, _, _, _ = model(input_ids=input_ids, attention_mask=mask, token_type_ids=token_type_ids, labels=targets, device=device)
        test_loss += loss_main.item()
        prob_lst.extend(main_prob)
        nb_test_steps += 1
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        if data == 'hans':
            predicted_labels = torch.where((predicted_labels == 0) | (predicted_labels == 1), torch.ones_like(predicted_labels), predicted_labels)
            pred = [hans_dict[label] for label in predicted_labels]
        else:
            pred = [mnli_dict[label] for label in predicted_labels]
        pred_lst.extend(pred)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        test_accuracy += tmp_test_accuracy

    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst, prob_lst


def generate_prediction_file(pred_lst, output_file):
    """Generate a prediction file from the list of predictions.

    Args:
        pred_lst (list): List of predictions.
        output_file (str): Path to the output file.
    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')


def generate_prob_file(prob_lst, prob_output_file_path):
    """Generate a probability file from the list of probabilities.

    Args:
        prob_lst (list): List of probabilities.
        prob_output_file_path (str): Path to the output file.
    """
    with open(prob_output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')


def main():
    """Main function to run the inference and generate prediction and probability files.

    Returns:
        None
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
    
    paws_output_file_path = os.path.join(input_path,'Predictions', 'PAWS', 'pred.txt')

    generate_prediction_file(paws_pred_lst, paws_output_file_path)
    generate_prob_file(paws_prob_lst, './Predictions/PAWS/prob_paws.txt')

    generate_prediction_file(train_pred_lst, './Predictions/train_pred.txt')
    generate_prob_file(train_prob_lst, './Predictions/train_prob.txt')

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

if __name__ == '__main__':
    main()
