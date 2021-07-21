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
    parser.add_argument('-rank', '--ranking_by', type=str, default=None,
                        required=False, help='Ranking candidates by which column')
    parser.add_argument('-rev', '--ranking_reverse', action='store_true', help='If set, ranking from high to low')

    args = parser.parse_args()
    return args


def full_Cartesian_RE(seed_concepts_path,
                      seed_relations_path,
                      concept_knn_path,
                      relation,
                      ranking_by,
                      ranking_reverse,
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
    cand_heads = cand_heads_df['neighbor'].tolist()
    cand_tails = cand_tails_df['neighbor'].tolist()
    
    if ranking_by is None:
        heads = [(_h, 0.0) for _h in seed_heads] + [(_h, 0.0) for _h in cand_heads if _h not in seed_heads]
        tails = [(_t, 0.0) for _t in seed_tails] + [(_t, 0.0) for _t in cand_tails if _t not in seed_tails]
    else:
        cand_h_pairs = list(zip(cand_heads, cand_heads_df[ranking_by].tolist()))
        cand_h_pairs.sort(key=lambda p: p[1], reverse=ranking_reverse)
        cand_t_pairs = list(zip(cand_tails, cand_tails_df[ranking_by].tolist()))
        cand_t_pairs.sort(key=lambda p: p[1], reverse=ranking_reverse)
        ## TODO: better way to decide seed score?
        seed_h_score = 1.0 if ranking_by == 'sim' else cand_h_pairs[0][1]
        seed_t_score = 1.0 if ranking_by == 'sim' else cand_t_pairs[0][1]
        
        heads = [(_h, seed_h_score) for _h in seed_heads] + cand_h_pairs
        tails = [(_t, seed_t_score) for _t in seed_tails] + cand_t_pairs

    if topk is not None:
        heads = heads[:topk]
        tails = tails[:topk]
        
    print('heads:', list(zip(*heads))[0])
    print('tails:', list(zip(*tails))[0])
    
    out_rels = []
    for _h, _hs in heads:
        for _t, _ts in tails:
            out_rels.append({
                'head': _h, 'relation': relation, 'tail': _t,
                'head_score': _hs, 'tail_score': _ts,
                'overall_score': _hs * _ts
            })
    
    if ranking_by is not None:
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