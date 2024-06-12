"""
    Compare two sets of data and calculate the Jaccard similarity between them.

    Usage::

        python compare_sets.py --file1 <FILE_PATH_1> --file2 <FILE_PATH_2>

    Arguments::
    
        --file1 (str): Path to the first pickled file.
        --file2 (str): Path to the second pickled file.
"""
import pickle
import argparse

def main():
    """
    Compare two sets of data and calculate the Jaccard similarity between them.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True, help='Path to the first pickled file')
    parser.add_argument('--file2', type=str, required=True, help='Path to the second pickled file')
    args = parser.parse_args()
    
    with open(args.file1, 'rb') as f:
        data1 = pickle.load(f)
    
    with open(args.file2, 'rb') as f:
        data2 = pickle.load(f)
    
    print(f'Comparing files: {args.file1} and {args.file2}')
    
    # Calculate Jaccard similarity
    int_set_bb_btb = data1.intersection(data2)
    un_set_bb_btb = data1.union(data2)
    bb_btb = len(int_set_bb_btb) / len(un_set_bb_btb)
    bb_btb = round(bb_btb, 3)

    print(f'Jaccard similarity between the two sets: {bb_btb}')

if __name__ == '__main__':
    main()
