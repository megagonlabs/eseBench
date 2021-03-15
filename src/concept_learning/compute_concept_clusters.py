from tqdm import tqdm
import logging
import argparse

import numpy as np
import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--embedding path', type=str,
                        required=True, help='Path to embeddings')
    parser.add_argument('-c', '--thread_ct', type=int, default=4,
                        help='No. of threads')
    parser.add_argument('-ca', '--clustering_algorithm', type=str, default='hdbscan', required=True,
                        help='hdbscan or knn')
    parser.add_argument('-s', '--cluster_size', type=int, default=2, required=True,
                        help='cluster size for knn or min cluster size for hdbscan')
    parser.add_argument('-d', '--cluster_dest', type=str, required=True,
                        help='Path to clusters')

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    args.input_file = os.path.join(args.corpus_name, 'sent_segmentation.txt')
    args.embed_dest = os.path.join(args.corpus_name, 'BERTembed.txt')
    args.embed_num = os.path.join(args.corpus_name, 'BERTembednum.txt')




if __name__ == "__main__":
    main()
    corpusName = sys.argv[1]
    num_thread = int(sys.argv[2])


