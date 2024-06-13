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

from data_loader import load_qqp, load_paws
# from Generate_Embeddings.Paraphrase_identification.Disentangled_feature_augmentation.data_loader import load_qqp, load_paws
import numpy

input_path = './'
input_model_path = './best_model'
paws_file_path = "../resources/PAWS/test.tsv"
BATCH_SIZE = 32



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
        self.num_labels = 2#config.num_labels
        self.model_l = AutoModel.from_pretrained("bert-base-uncased")
        self.model_b = AutoModel.from_pretrained("bert-base-uncased")
        self.fc_l_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_l_2 = nn.Linear(2*(self.num_labels), self.num_labels)
        self.fc_b_1 = nn.Linear(1536, 2*(self.num_labels))
        self.fc_b_2 = nn.Linear(2*(self.num_labels), self.num_labels)

    def features(self,input_ids, attention_mask):
        """
        Generate features using BERT models.

        Args:
            input_ids (torch.Tensor): The input token IDs tensor.
            attention_mask (torch.Tensor): The attention mask tensor.

        Returns:
            tuple: Tuple containing the features from the conflict BERT model and the alignment BERT model.

        """
        z_l = self.model_l(input_ids, attention_mask = attention_mask)
        z_b = self.model_b(input_ids, attention_mask = attention_mask)
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
        return output1,output2


def inference(model, dataloader, tokenizer, device, data = 'mnli'):
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

    mnli_dict = {0 : '0', 1 : '1'}
    for idx, batch in enumerate(tqdm(dataloader, ncols=100)):
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        nb_test_steps += 1
        with torch.no_grad():
            z_l, z_b = model.features(input_ids = input_ids, attention_mask =  mask)
            z = torch.cat((z_l,z_b), dim = 1)
            pred_conflict ,pred_align = model.linear(z_conflict = z, z_align = z)
            pred_prob = F.softmax(pred_conflict, dim = 1)
            prob_lst.extend(pred_prob)
            prob_lst.extend(pred_prob)
            predicted_labels = torch.argmax(pred_prob,dim=1)
            loss_dis_conflict = F.cross_entropy(pred_conflict, targets.view(-1), reduction='none')
            loss_dis_align = F.cross_entropy(pred_align,targets.view(-1), reduction = 'none')
            loss_weight = loss_dis_align / (loss_dis_align + loss_dis_conflict + 1e-8)
            loss = F.cross_entropy(pred_conflict, targets.view(-1))
            test_loss += loss
            targets = targets.view(-1)
            pred = [label for label in predicted_labels]
            pred_lst.extend(pred)
            tmp_test_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
            test_accuracy += tmp_test_accuracy
        weights.extend(loss_weight)
        print(idx,'print loss weight',loss_weight)
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

def main():
    """
    Main function to run evaluation on MNLI and HANS datasets.
    """
    gc.collect()

    torch.cuda.empty_cache()
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_model_path', type=str, required=True)
    parser.add_argument('--paws_test_path', type=str, required=True)
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.input_model_path)
    data = load_paws(file_path=args.paws_test_path, tokenizer=tokenizer)
    model = MainModel.from_pretrained(args.input_model_path, ignore_mismatched_sizes=True)
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(data.len)
    model.to(device)
    test_dataloader = DataLoader(data, shuffle = False, batch_size=BATCH_SIZE)
    print("Testing started")
    test_accuracy, paws_m_pred_lst, paws_m_prob_lst = inference(model, test_dataloader, tokenizer, device)
    print(f'\paws test accuracy: {test_accuracy}')
    
    output_file_path = os.path.join(input_path,'Predictions', 'PAWS', 'pred')
    
    generate_prediction_file(paws_m_pred_lst, output_file_path + '_paws_test.txt')
    
    generate_prob_file(paws_m_prob_lst, './Predictions/PAWS/prob_paws_test.txt')


    end = time.time()
    total_time = end - start
    with open('live-paws-testing.txt', 'a') as fh:
        fh.write(f'Total testing time : {total_time}\n')

    print(f"Total testing time : {total_time}")
if __name__ == '__main__':
    main()
