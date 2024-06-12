"""
This script calculates the Jaccard similarity coefficient between two sets of embeddings and identifies out-of-distribution (OOD) samples based on cosine similarity thresholds.

Usage::

    python knn_ood_v5.py --input_file_train <train_embedding_file> --train_model_name <model_name> --input_file_val <validation_embedding_file> --input_file2 <test_embedding_file> --pred_dir <prediction_directory> --test_groundtruth <test_groundtruth_file> --train_groundtruth <train_groundtruth_file> --percent <threshold_percentage>

Arguments::

    --input_file_train ( str ) : Path to the training embeddings file.
    --train_model_name ( str ) : Name of the training model.
    --input_file_val ( str ) : Path to the validation embeddings file.
    --input_file2 ( str ) : Path to the test embeddings file.
    --pred_dir ( str ) : Prediction directory.
    --test_groundtruth ( str ) : Path to the test ground truth file.
    --train_groundtruth ( str ) : Path to the train ground truth file.
    --percent ( float ) : Threshold percentage for identifying OOD samples based on cosine similarity.

Returns::

    Array containing the indices of OOD samples.

Example::

    python knn_ood_v5.py --input_file_train train_embeddings.txt --train_model_name Model1 --input_file_val validation_embeddings.txt --input_file2 test_embeddings.txt --pred_dir predictions --test_groundtruth test_groundtruth.txt --train_groundtruth train_groundtruth.txt --percent 95
    """

import numpy as np
import argparse
import os
import pickle
from sklearn.neighbors import NearestNeighbors

def read_data_func(dataset_name):
    """
    Read data from a file.
    
    Args:
    dataset_name (str): Path to the dataset file.
    
    Returns:
    numpy.ndarray: Data read from the file.
    """
    tensor = np.loadtxt(dataset_name)
    return tensor

def normalize_embedding_func(tensor):
    """
    Normalize embeddings by dividing each element by its row's sum.

    Args:
    tensor (numpy.ndarray): Input tensor.

    Returns:
    numpy.ndarray: Normalized tensor.
    """
    row_sq = np.square(tensor)
    row_sums = np.sum(row_sq, axis=1, keepdims=True)
    row_sums_sqrt = np.sqrt(row_sums)
    result_tensor = tensor / row_sums_sqrt
    return result_tensor

def select_threshold_func(tensor, percent):
    """
    Select a threshold value based on a percentage of the sorted tensor.

    Args:
    tensor (numpy.ndarray): Input tensor.
    percent (float): Percentage value.

    Returns:
    float: Threshold value.
    """
    tensor = np.sort(tensor)
    percent = percent/100
    n = len(tensor)
    percent = n * percent
    percent = int(percent)
    threshold = tensor[percent]
    return threshold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_train', type=str, required=True)
    parser.add_argument('--train_model_name', type=str, required=True)
    parser.add_argument('--input_file_val', type=str, required=True)
    parser.add_argument('--input_file2', type=str, required=True)
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--test_groundtruth', type=str, required=True)
    parser.add_argument('--train_groundtruth', type=str, required=True)
    parser.add_argument('--percent', type=float, required=True)
    args = parser.parse_args()

    tensor1_train = read_data_func(args.input_file_train)
    tensor1_val = read_data_func(args.input_file_val)
    tensor2 = read_data_func(args.input_file2)
    train_groundtruth = read_file(args.train_groundtruth)
    test_groundtruth = read_file(args.test_groundtruth)
    
    k_values = [10, 20, 50, 100, 200, 500, 800, 1000]
    
    tensor1_train = normalize_embedding_func(tensor1_train)
    tensor1_val = normalize_embedding_func(tensor1_val)
    tensor2 = normalize_embedding_func(tensor2)
    
    print(f'Train embeddings shape: {tensor1_train.shape}')
    print(f'Validation embeddings shape: {tensor1_val.shape}')
    print(f'Test embeddings shape: {tensor2.shape}')
    
    knn = NearestNeighbors(n_neighbors=1000, algorithm='brute', metric='cosine')
    knn = knn.fit(tensor1_train)
    distance, indices = knn.kneighbors(tensor1_val)
    test_distance, test_indices = knn.kneighbors(tensor2)

    for k in k_values:
        result_dir = './results/' + args.train_model_name + '/k_' + str(k) + '/' + args.pred_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        print(f'k = {k}')
        kth_values = distance[:, k-1]
        print(f'Validation kth values shape: {kth_values.shape}')
        kth_values = list(kth_values)
        
        print(f'Max cosine similarity value: {max(kth_values)}')
        print(f'Min cosine similarity value: {min(kth_values)}')
        threshold = select_threshold_func(kth_values, args.percent)
        print(f"Threshold: {threshold}")
        kth_values_test = test_distance[:, k-1]
        print(f'Test kth values shape: {kth_values_test.shape}')
        kth_values_test = list(kth_values_test)
        
        cnt = 0
        true_ind = []
        for i in range(len(test_indices)):
            groundtruth_i = test_groundtruth[i]
            dict = {}
            for j, index in enumerate(test_indices[i]):
                if j == k:
                    break
                if dict.get(train_groundtruth[index]) is None:
                    dict[train_groundtruth[index]] = 1
                else:
                    dict[train_groundtruth[index]] += 1
            max_val = 0
            max_val_class = ''
            for key, value in dict.items():
                if value > max_val:
                    max_val = value
                    max_val_class = key
            if max_val_class == 'contradiction' or max_val_class == 'neutral':
                max_val_class = 'non-entailment'

            if max_val_class != groundtruth_i:
                true_ind.append(i)
                cnt += 1

        true_ind = set(true_ind)
        print(f'Number of OOD data points: {len(true_ind)}')
        print(cnt)
        
        # Save the array with Pickle
        file_name = result_dir + '/' + args.pred_dir + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(true_ind, f)

        # Load the array with Pickle
        with open(file_name, 'rb') as f:
            loaded_array = pickle.load(f)

        print(len(loaded_array))
    
if __name__ == '__main__':
    main()
