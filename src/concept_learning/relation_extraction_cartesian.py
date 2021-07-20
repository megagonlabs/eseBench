from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cosine
from scipy.stats import gmean
import numpy as np
import pandas as pd
import os
import torch
import math
import json
from collections import defaultdict
from annoy import AnnoyIndex
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertTokenizer, BertModel, BertForMaskedLM

from utils import load_seed_aligned_concepts, load_seed_aligned_relations, load_embeddings

from roberta_ses.interface import Roberta_SES_Entailment

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-cknn', '--concept_knn_path', type=str,
                        required=True, help='Path to concept knn file')
    parser.add_argument('-r', '--relation', type=str, required=True,
                        help='The relation to extract for')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to output file')
#     parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
#                         help='embedding_dim')
    parser.add_argument('-topk', '--topk', type=int, default=None,
                        help='The number of entities to keep for each base head/tail')

    args = parser.parse_args()
    return args


def full_Cartesian_RE(seed_concepts_path,
                      seed_relations_path,
                      concept_knn_path,
                      relation,
                      topk=None,
                      dest=None,
                      **kwargs):
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_relations_df = pd.read_csv(seed_relations_path)
    relation_row = seed_relations_df[seed_relations_df['alignedRelationName'] == relation].iloc[0]
    concept_knn_results = pd.read_csv(concept_knn_path)
    
    head_type = relation_row['domain']
    tail_type = relation_row['range']
    print(head_type, '\t', tail_type)
    seed_heads = seed_concepts_df[seed_concepts_df['alignedCategoryName'] == head_type]['seedInstances'].item()
#     seed_heads = eval(list(seed_heads)[0])
    seed_tails = seed_concepts_df[seed_concepts_df['alignedCategoryName'] == tail_type]['seedInstances'].item()
#     seed_tails = eval(list(seed_tails)[0])
    print('seed_heads:', seed_heads)
    print('seed_tails:', seed_tails)

    # Candidate heads / tails from concept knn 
    cand_heads_df = concept_knn_results[concept_knn_results['concept'] == head_type]
    cand_tails_df = concept_knn_results[concept_knn_results['concept'] == tail_type]
    cand_heads = [(_h, 1.0) for _h in seed_heads] + \
        list(zip(cand_heads_df['neighbor'].tolist(), cand_heads_df['sim'].tolist()))
    cand_tails = [(_t, 1.0) for _t in seed_tails] + \
        list(zip(cand_tails_df['neighbor'].tolist(), cand_tails_df['sim'].tolist()))

    if topk is not None:
        cand_heads = cand_heads[:topk]
        cand_tails = cand_tails[:topk]
        
    print('cand_heads:', list(zip(*cand_heads))[0])
    print('cand_tails:', list(zip(*cand_tails))[0])
    
    out_rels = []
    for _h, _hs in cand_heads:
        for _t, _ts in cand_tails:
            out_rels.append({
                'head': _h, 'relation': relation, 'tail': _t,
                'overall_score': _hs * _ts
            })
    out_rels.sort(key=lambda d : d['overall_score'], reverse=True)
    
    out_rels_df = pd.DataFrame(out_rels)
    if dest is not None:
        out_rels_df.to_csv(dest, index=False)
    return out_rels_df

    
def main():
    args = parse_arguments()
    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
#     args.emb_path = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
#     args.templates_path = 'templates_manual.json'
    
#     args.dest = os.path.join(args.dataset_path, 'rel_extraction.csv')

    full_Cartesian_RE(**vars(args))
    
    
if __name__ == "__main__":
    main()