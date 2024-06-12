"""
Script for comparing two sets of embeddings using k-nearest neighbors (k-NN) approach.

This script imports necessary libraries including numpy, argparse, torch, faiss, matplotlib.pyplot, scipy.stats, sklearn, csv, and os. It defines functions to read data from files, normalize embeddings, calculate distances, sort distance matrices, select threshold values, and count out-of-distribution (OOD) samples. The main function loads embeddings from two files, normalizes them, computes distances using cosine similarity, finds k-nearest neighbors, and compares the sets of embeddings.

Attributes:
    numpy (module): Library for numerical computing.
    argparse (module): Library for parsing command-line arguments.
    torch (module): PyTorch library for deep learning.
    faiss (module): Library for efficient similarity search and clustering of dense vectors.
    metrics (module): User-defined module containing utility functions.
    pickle (module): Library for serializing and deserializing Python objects.
    matplotlib.pyplot (module): Library for creating visualizations in Python.
    scipy.stats (module): Library for statistics and probability distributions.
    sklearn (module): Library for machine learning algorithms.
    csv (module): Library for reading and writing CSV files.
    os (module): Library for interacting with the operating system.
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

def read_data_func(dataset_name):
    """Reads embeddings from a file.
    
    Args:
        dataset_name (str): The path to the file containing embeddings.
        
    Returns:
        numpy.ndarray: A NumPy array containing the embeddings.
    """
    tensor = np.loadtxt(dataset_name, dtype=float, delimiter=' ')
    return tensor

def normalize_embedding_func(tensor):
    """Normalizes the embeddings.
    
    Args:
        tensor (numpy.ndarray): A NumPy array containing the embeddings.
        
    Returns:
        numpy.ndarray: A NumPy array containing the normalized embeddings.
    """
    row_sq = np.square(tensor)
    row_sums = np.sum(row_sq, axis=1, keepdims=True)
    row_sums_sqrt = np.sqrt(row_sums)
    result_tensor = tensor / row_sums_sqrt
    return result_tensor

def find_distance_func(tensor1, tensor2):
    """Finds the pairwise distances between two sets of embeddings.
    
    Args:
        tensor1 (numpy.ndarray): A NumPy array containing the first set of embeddings.
        tensor2 (numpy.ndarray): A NumPy array containing the second set of embeddings.
        
    Returns:
        numpy.ndarray: A 2D NumPy array containing the pairwise distances.
    """
    n = len(tensor1)
    m = len(tensor2)
    distance_matrix = []
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
    """Sorts the distance matrix by row.
    
    Args:
        distance_matrix (numpy.ndarray): A 2D NumPy array containing the distance matrix.
        
    Returns:
        numpy.ndarray: A 2D NumPy array containing the sorted distance matrix.
    """
    distance_matrix.sort(axis=1)
    return distance_matrix

def pick_kth_value_func(distance_matrix, k):
    """Picks the k-th value from each row of the distance matrix.
    
    Args:
        distance_matrix (numpy.ndarray): A 2D NumPy array containing the distance matrix.
        k (int): The index of the value to pick from each row.
        
    Returns:
        numpy.ndarray: A 1D NumPy array containing the k-th values from each row.
    """
    tensor = distance_matrix[:, k]
    return tensor

def select_threshold_func(tensor, percent):
    """Selects a threshold value based on the given percentile.
    
    Args:
        tensor (numpy.ndarray): A 1D NumPy array containing distance values.
        percent (float): The percentile value (0-100) to select the threshold.
        
    Returns:
        float: The threshold value.
    """
    tensor = np.sort(tensor)
    percent = percent / 100
    n = len(tensor)
    percent = n * percent
    percent = int(percent)
    threshold = tensor[percent]
    return threshold

def get_ood_count_func(tensor, th, val):
    """Counts the number of out-of-distribution (OOD) samples based on a threshold.
    
    Args:
        tensor (numpy.ndarray): A 1D NumPy array containing distance values.
        th (float): The threshold value.
        val (int): The total number of samples.
        
    Returns:
        numpy.ndarray: A 1D NumPy array containing the OOD counts for each sample.
    """
    mask = tensor > th
    count_per_row = np.sum(mask, axis=1)
    count_per_row = count_per_row / val
    return count_per_row

def main():
    """Main function for comparing two sets of embeddings using k-NN approach."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    args = parser.parse_args()

    # Loading the embeddings
    model1 = read_data_func(args.file1)
    model2 = read_data_func(args.file2)

    # Normalize the embeddings
    model1 = normalize_embedding_func(model1)
    model2 = normalize_embedding_func(model2)

    n_neighbors = min(len(model1), 1000)
    k_values = [10, 20, 50, 100, 200, 500, 800, 1000]
    model1_model2 = []
    knn1 = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')
    knn2 = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric='cosine')

    knn1 = knn1.fit(model1)
    distance_model1, indices_model1 = knn1.kneighbors(model1)

    knn2 = knn2.fit(model2)
    distance_model2, indices_model2 = knn2.kneighbors(model2)

    for k in k_values:
        if k > n_neighbors:
            break
        model1_ind = indices_model1[:, :k - 1]
        model2_ind = indices_model2[:, :k - 1]

        s_model1_model2 = 0.0

        for i in range(len(model1_ind)):
            a = len(set(model1_ind[i]).intersection(set(model2_ind[i])))
            b = len(set(model1_ind[i]).union(set(model2_ind[i])))
            s_model1_model2 = s_model1_model2 + (a / b)

        print(f"Result for K = {k} : ")

        print(s_model1_model2/len(model1_ind))  

		
     			
if __name__ == '__main__':
    main()
