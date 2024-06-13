"""
This script test the model trained for Natural Language Inference(NLI) task. It saves the prediction file and the softmax score file

Usage::

    python test.py --input_model_path <path_to_pretrained_model> --mnli_path <path_to_mnli_train_file> --mnli_val_path <path_to_mnli_val> --hans_test_path <hans_dataset_path>

Arguments::

    --input_model_path (str): Path to the pretrained model directory.
    --mnli_train_path (str): Path to the MNLI train dataset file.
    --mnli_val_path (str): Path to the MNLI validation dataset file.
    --hans_test_path (str): Path to the HANS test file.
"""
import torch
from tqdm import tqdm
import time
import pickle
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
from Generate_Embeddings.NLI.Vanilla.data_loader import load_mnli
from Generate_Embeddings.NLI.Vanilla.data_loader import load_hans
from Generate_Embeddings.NLI.Vanilla.data_loader import load_snli

input_path = './'
input_model_path = './best_model'
mnli_file1_path = './multinli_1.0/multinli_1.0_dev_matched.txt'
mnli_file2_path = './multinli_1.0/multinli_1.0_dev_mismatched.txt'
hans_file1_path = './HANS/hans1.txt'
hans_file2_path = './HANS/hans2.txt'
BATCH_SIZE = 32

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


def inference(model, dataloader, tokenizer, device, data = 'mnli'):
    """
    Perform inference on the provided data using the given model.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for processing text data.
        device (str): Device to perform computation ('cuda' or 'cpu').
        data (str): Dataset used for testing. Default is 'mnli'.

    Returns:
        tuple: Test accuracy, list of predicted labels, list of probabilities.
    """
    model.eval()
    prob_lst = []
    pred_lst = []
    test_loss = 0
    bias_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    hans_dict = {1 : 'non-entailment', 2 : 'entailment'}
    mnli_dict = {0 : 'contradiction', 1 : 'neutral', 2 : 'entailment'}
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        with torch.no_grad():
            loss_main,main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets, device = device)
        test_loss += loss_main.item()
        nb_test_steps += 1
        prob_lst.extend(main_prob)
        predicted_labels = torch.argmax(main_prob, dim=1)
        targets = targets.view(-1)
        if data == 'hans':
            predicted_labels = torch.where((predicted_labels == 0) | (predicted_labels == 1), torch.ones_like(predicted_labels), predicted_labels)
            pred = [hans_dict[label.item()] for label in predicted_labels]
        else:
            pred = [mnli_dict[label.item()] for label in predicted_labels]
        pred_lst.extend(pred)
        tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        test_accuracy += tmp_test_accuracy
        
    test_accuracy = test_accuracy / nb_test_steps
    
    return test_accuracy, pred_lst, prob_lst


def generate_prediction_file(pred_lst, output_file):
    """
    Generate a file containing predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file_path (str): Path to the output file.
    
    Returns:
        None
    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')

def generate_prob_file(prob_lst, prob_output_file_path):
    """
    Generate a file containing prediction probabilities.

    Args:
        prob_lst (list): List of probabilities for each prediction.
        output_file_path (str): Path to the output file.
    
    Returns:
        None
    """
    with open(prob_output_file_path, 'w') as fh:
        for probs in prob_lst:
            for prob in probs:
                fh.write(f'{prob} ')
            fh.write('\n')


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



def main():
    """
    Main function to run the model training and inference pipeline.
    """
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--mnli_train_path', type=str, required=True)
    parser.add_argument('--mnli_val_path', type=str, required=True)
    parser.add_argument('--hans_test_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    model = MainModel.from_pretrained(args.input_model_path, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    
    model.to(device)
    print("Testing started")
    #loding HANS data
    data1 = load_hans(file_path=args.hans_test_path, tokenizer=tokenizer)
    hans_data = data1
    mnli_test_data = read_dataset('../resources/multinli_1.0/test.pkl')
    mnli_val_data = read_dataset(args.mnli_val_path)

    train_data = load_mnli(file_path=args.mnli_train_path, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data, shuffle = False, batch_size=BATCH_SIZE)
    snli_data = load_snli(file_path='../resources/snli_1.0/snli_1.0_test.txt', tokenizer=tokenizer)
    hans_test_dataloader = DataLoader(hans_data, shuffle = False, batch_size=BATCH_SIZE)
    mnli_test_dataloader = DataLoader(mnli_test_data, shuffle = False, batch_size=BATCH_SIZE)
    mnli_val_dataloader = DataLoader(mnli_val_data, shuffle = False, batch_size=BATCH_SIZE)
    snli_test_dataloader = DataLoader(snli_data, shuffle = False, batch_size=BATCH_SIZE)

    mnli_train_accuracy, mnli_train_pred_lst, mnli_train_prob_lst = inference(model, train_dataloader, tokenizer, device)
    print(f'\tMNLI train accuracy : {mnli_train_accuracy}')
    mnli_val_accuracy, mnli_val_pred_lst, mnli_val_prob_lst = inference(model, mnli_val_dataloader, tokenizer, device)
    print(f'\tMNLI val accuracy : {mnli_val_accuracy}')
    hans_test_accuracy, hans_pred_lst, hans_prob_lst = inference(model, hans_test_dataloader, tokenizer, device, data = 'hans')
    print(f'\tHANS set test accuracy: {hans_test_accuracy}')
    mnli_test_accuracy, mnli_test_pred_lst, mnli_test_prob_lst = inference(model, mnli_test_dataloader, tokenizer, device)
    print(f'\tMNLI test accuracy : {mnli_test_accuracy}')
    snli_test_accuracy, snli_pred_lst, snli_prob_lst = inference(model, snli_test_dataloader, tokenizer, device)
    print(f'\tSNLI set test accuracy: {snli_test_accuracy}')
    
    mnli_snli_data = ConcatDataset([mnli_test_data, snli_data])
    mnli_snli_test_dataloader = DataLoader(mnli_snli_data, shuffle = False, batch_size=BATCH_SIZE)
    
    mnli_snli_test_accuracy, mnli_snli_pred_lst, mnli_snli_prob_lst = inference(model, mnli_snli_test_dataloader, tokenizer, device)
    print(f'\tMNLI U SNLI set test accuracy: {mnli_snli_test_accuracy}')
    
    

    output_file_path = os.path.join(input_path,'Predictions', 'MNLI', 'pred')
    hans_output_file_path = os.path.join(input_path,'Predictions', 'HANS', 'pred_hans.txt')
    
    generate_prediction_file(mnli_train_pred_lst, output_file_path + '_mnli_train.txt')
    generate_prediction_file(mnli_val_pred_lst, output_file_path + '_mnli_val.txt')
    generate_prediction_file(mnli_test_pred_lst, output_file_path + '_mnli_test.txt')
    generate_prediction_file(hans_pred_lst, hans_output_file_path)
    generate_prediction_file(snli_pred_lst, './Predictions/SNLI/pred_snli.txt')

    generate_prob_file(mnli_train_prob_lst, './Predictions/MNLI/prob_mnli_train.txt')
    generate_prob_file(mnli_val_prob_lst, './Predictions/MNLI/prob_mnli_val.txt')
    generate_prob_file(mnli_test_prob_lst, './Predictions/MNLI/prob_mnli_test.txt')
    generate_prob_file(hans_prob_lst, './Predictions/HANS/prob_hans.txt')
    generate_prob_file(snli_prob_lst, './Predictions/SNLI/prob_snli.txt')

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total testing time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
