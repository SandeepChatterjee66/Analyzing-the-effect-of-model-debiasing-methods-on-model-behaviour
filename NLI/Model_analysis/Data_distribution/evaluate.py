"""
This Python script compares two sets of data stored in pickle files and calculates the Jaccard similarity coefficient between them.

Usage::

    python compare_sets.py --file1 <path_to_pickle_file1> --file2 <path_to_pickle_file2>

Arguments::

    --file1 ( str ) : The file path to the first pickle file containing data.
    --file2 ( str ) : The file path to the second pickle file containing data.

Returns::

    Jaccard similarity coefficient (float) between the two sets of data.

Example::

    python compare_sets.py --file1 data1.pkl --file2 data2.pkl
"""

import pickle
import csv
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True)
    parser.add_argument('--file2', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.file1, 'rb') as f:
        data1 = pickle.load(f)
    
    with open(args.file2, 'rb') as f:
        data2 = pickle.load(f)
    
    print(args.file1)
    print(args.file2)
    
    #print(data1)

    int_set_bb_btb = data1.intersection(data2)

    un_set_bb_btb = data1.union(data2)

    bb_btb = len(int_set_bb_btb)/len(un_set_bb_btb)

    bb_btb = round(bb_btb,3)

    print(bb_btb)
	

if __name__ == '__main__':
	main()