"""
This script evaluates the similarity between two sets of embeddings using the K-nearest neighbors (KNN) approach.

Usage ::

    python3 evaluate_similarity.py --file1 <path_to_file1> --file2 <path_to_file2>

Arguments ::

    --file1: Path to the first set of embeddings file.
    
    --file2: Path to the second set of embeddings file.
"""


import numpy as np
import argparse
import torch
import faiss
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import csv
import os

"""This script contains several functions to evaluate the similarity between two sets of embeddings using the K-nearest neighbors (KNN) approach.

Function Documentation:

1. read_data_func(dataset_name):
    - Description: Reads the embeddings from the specified dataset file.
    - Arguments:
        - dataset_name: Path to the dataset file containing embeddings.
    - Returns:
        - tensor: Numpy array containing the embeddings.

2. normalize_embedding_func(tensor):
    - Description: Normalizes the embeddings to ensure consistency in magnitude.
    - Arguments:
        - tensor: Numpy array containing the embeddings.
    - Returns:
        - result_tensor: Normalized embeddings.

3. find_distance_func(tensor1, tensor2):
    - Description: Computes the cosine distance matrix between two sets of embeddings.
    - Arguments:
        - tensor1: Numpy array containing the embeddings of the first model.
        - tensor2: Numpy array containing the embeddings of the second model.
    - Returns:
        - distance_matrix: 2D numpy array representing the cosine distance matrix.

4. sort_distance_matrix_func(distance_matrix):
    - Description: Sorts the distance matrix along each row.
    - Arguments:
        - distance_matrix: 2D numpy array representing the cosine distance matrix.
    - Returns:
        - sorted_distance_matrix: 2D numpy array with sorted distances along each row.

5. pick_kth_value_func(distance_matrix, k):
    - Description: Picks the k-th value from each row of the distance matrix.
    - Arguments:
        - distance_matrix: 2D numpy array representing the cosine distance matrix.
        - k: Integer value representing the index of the value to be picked.
    - Returns:
        - tensor: Numpy array containing the k-th values from each row.

6. select_threshold_func(tensor, percent):
    - Description: Selects a threshold value based on the specified percentile of the values.
    - Arguments:
        - tensor: Numpy array containing the k-th values from each row.
        - percent: Integer representing the percentile value (0-100).
    - Returns:
        - threshold: Threshold value selected based on the specified percentile.

7. get_ood_count_func(tensor, th, val):
    - Description: Computes the count of out-of-distribution (OOD) points based on a threshold.
    - Arguments:
        - tensor: Numpy array containing the k-th values from each row.
        - th: Threshold value to determine OOD points.
        - val: Total number of values considered in the computation.
    - Returns:
        - count_per_row: Numpy array containing the count of OOD points per row.

"""

def read_data_func(dataset_name):
    """
    Read embeddings from the specified dataset file.

    Parameters:
    - dataset_name (str): Path to the dataset file containing embeddings.

    Returns:
    - tensor (numpy.ndarray): Numpy array containing the embeddings.
    """
    tensor = np.loadtxt(dataset_name, dtype=float, delimiter=' ')
    return tensor

def normalize_embedding_func(tensor):
    """
    Normalize embeddings to ensure consistency in magnitude.

    Parameters:
    - tensor (numpy.ndarray): Numpy array containing the embeddings.

    Returns:
    - result_tensor (numpy.ndarray): Normalized embeddings.
    """
    # Calculate the sum of each row
    row_sq = np.square(tensor)
    row_sums = np.sum(row_sq, axis=1, keepdims=True)
    row_sums_sqrt = np.sqrt(row_sums)

    # Divide each element by its row's sum
    result_tensor = tensor / row_sums_sqrt

    return result_tensor

def find_distance_func(tensor1, tensor2):
    """
    Compute the cosine distance matrix between two sets of embeddings.

    Parameters:
    - tensor1 (numpy.ndarray): Numpy array containing the embeddings of the first model.
    - tensor2 (numpy.ndarray): Numpy array containing the embeddings of the second model.

    Returns:
    - distance_matrix (numpy.ndarray): 2D numpy array representing the cosine distance matrix.
    """
    # Getting the length of the tensor
    n = len(tensor1)
    m = len(tensor2)
    distance_matrix = []

    # For every row, calculate the distance to every row
    for i in range(n):
        distance_row = []
        for j in range(m):
            distance_tensor = np.subtract(tensor1[i], tensor2[j])
            distance_tensor_sq = np.square(distance_tensor)
            dist_square = np.sum(distance_tensor_sq, axis=0)
            distance = np.sqrt(dist_square)
            distance_row.append(distance)
        distance_matrix.append(distance_row)

    distance_matrix = np.array(distance_matrix)
    return distance_matrix

def sort_distance_matrix_func(distance_matrix):
    """
    Sort the distance matrix along each row.

    Parameters:
    - distance_matrix (numpy.ndarray): 2D numpy array representing the cosine distance matrix.

    Returns:
    - sorted_distance_matrix (numpy.ndarray): 2D numpy array with sorted distances along each row.
    """
    distance_matrix.sort(axis=1)
    return distance_matrix

def pick_kth_value_func(distance_matrix, k):
    """
    Pick the k-th value from each row of the distance matrix.

    Parameters:
    - distance_matrix (numpy.ndarray): 2D numpy array representing the cosine distance matrix.
    - k (int): Index of the value to be picked.

    Returns:
    - tensor (numpy.ndarray): Numpy array containing the k-th values from each row.
    """
    # Pick the k-th column from 2D matrix
    tensor = distance_matrix[:, k]
    return tensor

def select_threshold_func(tensor, percent):
    """
    Select a threshold value based on the specified percentile of the values.

    Parameters:
    - tensor (numpy.ndarray): Numpy array containing the k-th values from each row.
    - percent (int): Percentile value (0-100).

    Returns:
    - threshold (float): Threshold value selected based on the specified percentile.
    """
    tensor = np.sort(tensor)
    percent = percent / 100
    n = len(tensor)
    percent = n * percent
    percent = int(percent)
    threshold = tensor[percent]
    return threshold

def get_ood_count_func(tensor, th, val):
    """
    Compute the count of out-of-distribution (OOD) points based on a threshold.

    Parameters:
    - tensor (numpy.ndarray): Numpy array containing the k-th values from each row.
    - th (float): Threshold value to determine OOD points.
    - val (int): Total number of values considered in the computation.

    Returns:
    - count_per_row (numpy.ndarray): Numpy array containing the count of OOD points per row.
    """
    # Create a boolean mask for elements greater than the threshold
    mask = tensor > th

    # Count the number of elements greater than the threshold in each row
    count_per_row = np.sum(mask, axis=1)
    count_per_row = count_per_row / val

    return count_per_row



def main():

    """
    Script Workflow:

    1. Read Data:
        - Reads the embeddings from the provided files.
        
    2. Normalize Embeddings:
        - Normalizes the embeddings to ensure consistency in magnitude.

    3. Compute Distance Matrix:
        - Calculates the cosine distance matrix between the embeddings from both models.

    4. K-nearest Neighbors (KNN):
        - Performs KNN search to find the nearest neighbors for each embedding in both models.

    5. Similarity Calculation:
        - Calculates the similarity between the nearest neighbors of the two models for each embedding.

    6. Output:
        - Prints the similarity score for each specified value of k.

    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)

        


    args = parser.parse_args()

    # loading the embeddings 
    model1 = read_data_func(args.file1)
    model2 = read_data_func(args.file2)

    print(len(model1))
    print(len(model2))
    # normalize the embeddings
    model1 = normalize_embedding_func(model1)
    model2 = normalize_embedding_func(model2)


    k_values=[10,20,50,100,200,500,800,1000]
    model1_model2 = []
    knn1 = NearestNeighbors(n_neighbors=1000, algorithm='brute', metric='euclidean')

    knn2 = NearestNeighbors(n_neighbors=1000, algorithm='brute', metric='euclidean')
    knn1 = knn1.fit(model1)
    distance_model1, indices_model1 = knn1.kneighbors(model1)

    knn2 = knn2.fit(model2)
    distance_model2, indices_model2 = knn2.kneighbors(model2)

    for k in k_values: 
        model1_ind = indices_model1[:,:k-1]
        model2_ind = indices_model2[:,:k-1]


        s_model1_model2 = 0.0

        for i in range(len(model1_ind)):
            a = len(set(model1_ind[i]).intersection(set(model2_ind[i])))
            b = len(set(model1_ind[i]).union(set(model2_ind[i])))
            
            s_model1_model2 = s_model1_model2 + (a/b)

        print(f"Result for K = {k} : ")
        print(s_model1_model2/len(model1_ind))  

		
     			
if __name__ == '__main__':
    main()
