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
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import os
import gc
import argparse
import pickle
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from Generate_Embeddings.NLI.Disentangled_feature_augmentation.data_loader import *

input_path = './'
input_model_path = './best_model'
mnli_file1_path = './multinli_1.0/multinli_1.0_dev_matched.txt'
mnli_file2_path = './multinli_1.0/multinli_1.0_dev_mismatched.txt'
hans_file1_path = './HANS/hans1.txt'
hans_file2_path = './HANS/hans2.txt'
BATCH_SIZE = 36

class GeneralizedCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss.

    This loss function computes the generalized cross entropy loss given the logits
    and the target labels. It is defined as follows:
    
    Loss = -∑(pk^q * log(pk)) / q

    Args:
        q (float): The power parameter for the Generalized Cross Entropy Loss. 
            Default is 0.7.

    Returns:
        tensor: The computed loss value.

    Raises:
        NameError: If the computed probabilities or targets contain NaN values.

    """

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        """
        Forward pass of the Generalized Cross Entropy Loss.

        Args:
            logits (tensor): The logits tensor from the model.
            targets (tensor): The target labels tensor.

        Returns:
            tensor: The computed loss tensor.

        Raises:
            NameError: If the computed probabilities or targets contain NaN values.

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
    Main Model incorporating two BERT models.

    This model combines two BERT models, one for conflict detection and the other for alignment detection.
    It generates features using each BERT model and then passes them through linear layers for classification.

    Args:
        config (BertConfig): The configuration object.

    Attributes:
        num_labels (int): The number of labels.
        model_l (BertModel): BERT model for conflict detection.
        model_b (BertModel): BERT model for alignment detection.
        fc_l_1 (nn.Linear): Linear layer for conflict detection.
        fc_l_2 (nn.Linear): Linear layer for conflict detection.
        fc_b_1 (nn.Linear): Linear layer for alignment detection.
        fc_b_2 (nn.Linear): Linear layer for alignment detection.

    """

    def __init__(self, config):
        super(MainModel,self).__init__(config)
        self.num_labels = config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_l_2 = nn.Linear(2*(self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_b_2 = nn.Linear(2*(self.num_labels), self.num_labels)

    def features(self, input_ids, attention_mask):
        """
        Generate features using BERT models.

        Args:
            input_ids (torch.Tensor): The input token IDs tensor.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            tuple: Tuple containing the features from the conflict BERT model and the alignment BERT model.

        """
        z_l = self.model_l(input_ids, attention_mask=attention_mask)
        z_b = self.model_b(input_ids, attention_mask=attention_mask)
        return z_l.last_hidden_state[:,0,:], z_b.last_hidden_state[:,0,:]

    def linear(self, z_conflict, z_align):
        """
        Pass features through linear layers for classification.

        Args:
            z_conflict (torch.Tensor): Features from the conflict BERT model.
            z_align (torch.Tensor): Features from the alignment BERT model.

        Returns:
            tuple: Tuple containing the output tensors from the conflict linear layers and the alignment linear layers.

        """
        hidden_output1 = self.fc_l_1(z_conflict)
        output1 = self.fc_l_2(hidden_output1)
        hidden_output2 = self.fc_b_1(z_align)
        output2 = self.fc_b_2(hidden_output2)
        return output1, output2



def inference(model, dataloader, tokenizer, device, data='mnli'):
    """
    Perform inference on the provided data using the model.

    Args:
        model (MainModel): The trained model for inference.
        dataloader (DataLoader): The DataLoader containing the data for inference.
        tokenizer: The tokenizer used for tokenizing the input data.
        device (str): The device to perform inference on (e.g., 'cpu' or 'cuda').
        data (str, optional): Specifies the type of data ('mnli' or 'hans'). Defaults to 'mnli'.

    Returns:
        tuple: A tuple containing the accuracy, predicted labels, and probabilities.

    """
    model.eval()
    pred_lst = []
    weights = []
    prob_lst = []
    test_loss = 0
    nb_test_steps = 0
    test_accuracy = 0
    hans_dict = {1: 'non-entailment', 2: 'entailment'}
    mnli_dict = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)

        nb_test_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids=input_ids, attention_mask=mask)
            z = torch.cat((z_l, z_b), dim=1)
            pred_conflict, pred_align = model.linear(z_conflict=z, z_align=z)
            pred_prob = F.softmax(pred_conflict, dim=1)
            prob_lst.extend(pred_prob)
            prob_lst.extend(pred_prob)
            predicted_labels = torch.argmax(pred_prob, dim=1)
            loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
            loss_dis_align = F.cross_entropy(pred_align, targets.view(-1), reduction='none')
            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
            loss = F.cross_entropy(pred_conflict, targets.view(-1))
            test_loss += loss
            targets = targets.view(-1)
            if data == 'hans':
                predicted_labels = torch.where((predicted_labels == 0) | (predicted_labels == 1), torch.ones_like(predicted_labels), predicted_labels)
                pred = [hans_dict[label.item()] for label in predicted_labels]
            else:
                pred = [mnli_dict[label.item()] for label in predicted_labels]
            pred_lst.extend(pred)
            tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            test_accuracy += tmp_test_accuracy
        weights.extend(loss_weight)
    test_accuracy = test_accuracy / nb_test_steps
    return test_accuracy, pred_lst, prob_lst


def generate_prediction_file(pred_lst, output_file):
    """
    Generate a prediction file from a list of predictions.

    Args:
        pred_lst (list): List of predicted labels.
        output_file (str): Path to the output file.

    """
    with open(output_file, 'w') as fh:
        for pred in pred_lst:
            fh.write(f'{pred}\n')


def generate_prob_file(prob_lst, prob_output_file_path):
    """
    Generate a probability file from a list of probability lists.

    Args:
        prob_lst (list): List of probability lists.
        prob_output_file_path (str): Path to the output probability file.

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
    Main function to run evaluation on MNLI and HANS datasets.
    """
    gc.collect()
    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--mnli_matched_path', type=str, required=True)
    parser.add_argument('--mnli_mismatched_path', type=str, required=True)
    parser.add_argument('--hans_file1_path', type=str, required=True)
    parser.add_argument('--hans_file2_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    
    # Initialize model
    model = MainModel.from_pretrained(args.input_model_path)
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    print("Testing started")

    # Loading HANS data
    hans_data = load_hans(file_path=args.hans_test_path, tokenizer=tokenizer)
    mnli_test_data = read_dataset('../resources/multinli_1.0/test.pkl')
    mnli_val_data = read_dataset(args.mnli_val_path)
    train_data = load_mnli(file_path=args.mnli_train_path, tokenizer=tokenizer)

    # Dataloaders
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    snli_data = load_snli(file_path='../resources/snli_1.0/snli_1.0_test.txt', tokenizer=tokenizer)
    hans_test_dataloader = DataLoader(hans_data, shuffle=False, batch_size=BATCH_SIZE)
    mnli_test_dataloader = DataLoader(mnli_test_data, shuffle=False, batch_size=BATCH_SIZE)
    mnli_val_dataloader = DataLoader(mnli_val_data, shuffle=False, batch_size=BATCH_SIZE)
    snli_test_dataloader = DataLoader(snli_data, shuffle=False, batch_size=BATCH_SIZE)
    mnli_snli_data = ConcatDataset([mnli_test_data, snli_data])
    mnli_snli_test_dataloader = DataLoader(mnli_snli_data, shuffle=False, batch_size=BATCH_SIZE)

    # Inference and evaluation
    mnli_train_accuracy, mnli_train_pred_lst, mnli_train_prob_lst = inference(model, train_dataloader, tokenizer, device)
    print(f'\tMNLI train accuracy: {mnli_train_accuracy}')
    mnli_val_accuracy, mnli_val_pred_lst, mnli_val_prob_lst = inference(model, mnli_val_dataloader, tokenizer, device)
    print(f'\tMNLI val accuracy: {mnli_val_accuracy}')
    hans_test_accuracy, hans_pred_lst, hans_prob_lst = inference(model, hans_test_dataloader, tokenizer, device, data='hans')
    print(f'\tHANS set test accuracy: {hans_test_accuracy}')
    mnli_test_accuracy, mnli_test_pred_lst, mnli_test_prob_lst = inference(model, mnli_test_dataloader, tokenizer, device)
    print(f'\tMNLI test accuracy: {mnli_test_accuracy}')
    snli_test_accuracy, snli_pred_lst, snli_prob_lst = inference(model, snli_test_dataloader, tokenizer, device)
    print(f'\tSNLI set test accuracy: {snli_test_accuracy}')
    mnli_snli_test_accuracy, mnli_snli_pred_lst, mnli_snli_prob_lst = inference(model, mnli_snli_test_dataloader, tokenizer, device)
    print(f'\tMNLI U SNLI set test accuracy: {mnli_snli_test_accuracy}')

    # File generation
    output_file_path = os.path.join(input_path, 'Predictions', 'MNLI', 'pred')
    hans_output_file_path = os.path.join(input_path, 'Predictions', 'HANS', 'pred_hans.txt')
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
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")


if __name__ == '__main__':
    main()

