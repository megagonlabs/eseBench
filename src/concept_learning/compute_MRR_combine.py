from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd
import os
import torch
import math
from annoy import AnnoyIndex
from collections import defaultdict
from scipy.stats import pearsonr, entropy, gmean
import random


from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, get_masked_contexts
from lm_probes import LMProbe, LMProbe_GPT2, LMProbe_Joint

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--pred_1_path', type=str, required=True,
                        help='The EE prediction path of method 1')
    parser.add_argument('-p2', '--pred_2_path', type=str, required=True,
                        help='The EE prediction path of method 2')
    parser.add_argument('-r1', '--pred_1_rank', type=str, default=None,
                        help='The ranking metrics for method 1 (e.g. "sim" for emb); None (default): ranking already in file, same below')
    parser.add_argument('-r2', '--pred_2_rank', type=str, default=None,
                        help='The ranking metrics for method 2 (e.g. "lm_score" for LM)')
    parser.add_argument('-rev1', '--rev1', action='store_true',
                        help='Ranking pred 1 reversed (high-to-low), default is False')
    parser.add_argument('-rev2', '--rev2', action='store_true',
                        help='Ranking pred 2 reversed (high-to-low), default is False')
    parser.add_argument('-w1', '--weight_1', type=float, default=1.0,
                        help='The MRR combination weight of method 1')
    parser.add_argument('-w2', '--weight_2', type=float, default=1.0,
                        help='The MRR combination weight of method 2')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to output rankings')
    parser.add_argument('-topk', '--topk', type=int, default=None,
                        help='Top-k to extract for each concept; None (default): keep all entities')

    args = parser.parse_args()
    return args


def MRR_combine(pred_1_path,
                pred_2_path,
                pred_1_rank,
                pred_2_rank,
                rev1,
                rev2,
                weight_1,
                weight_2,
                dest,
                topk,
                **kwargs):
    
    ee_1_df = pd.read_csv(pred_1_path)
    ee_2_df = pd.read_csv(pred_2_path)
    concept_list = ee_1_df['concept'].drop_duplicates().tolist()
    _concept_list = ee_2_df['concept'].drop_duplicates().tolist()
    assert set(concept_list) == set(_concept_list)

    if pred_1_rank is None:
        pred_1_rank = 'rank_1'
        ee_1_df['rank_1'] = float('nan')
    if pred_2_rank is None:
        pred_2_rank = 'rank_2'
        ee_2_df['rank_2'] = float('nan')
    
    ## Using MRR to combine ranking 
    ee_mrr_combine_list = []

    for _cc in sorted(concept_list):
        _cc_df_1 = ee_1_df[ee_1_df['concept'] == _cc]
        if pred_1_rank is not None:
            _cc_df_1 = _cc_df_1.sort_values(by=pred_1_rank, ascending=not rev1)
        else:
            _cc_df_1['rank_1'] = range(1, len(_cc_df_1) + 1)
        # _ee_list_1 = _cc_df_1['neighbor'].tolist()
        _ee_list_1 = _cc_df_1['neighbor'].apply(lambda e: e.lower()).drop_duplicates().tolist()
        
        _cc_df_2 = ee_2_df[ee_2_df['concept'] == _cc]
        if pred_2_rank is not None:
            _cc_df_2 = _cc_df_2.sort_values(by=pred_2_rank, ascending=not rev2)
        else:
            _cc_df_2['rank_2'] = range(1, len(_cc_df_2) + 1)
        # _ee_list_2 = _cc_df_2['neighbor'].tolist()
        _ee_list_2 = _cc_df_2['neighbor'].apply(lambda e: e.lower()).drop_duplicates().tolist()

        _all_entities_mrr = defaultdict(float)
        for i, _e in enumerate(_ee_list_1):
            _all_entities_mrr[_e] += weight_1 / (i+1)
        for i, _e in enumerate(_ee_list_2):
            _all_entities_mrr[_e] += weight_2 / (i+1)

        _all_entities_mrr_list = sorted(list(_all_entities_mrr.items()), key=lambda p: p[-1], reverse=True)

        if topk is not None:
            _all_entities_mrr_list = _all_entities_mrr_list[:topk]

        for _e, _mrr in _all_entities_mrr_list:
            ee_mrr_combine_list.append((_cc, _e, _mrr))

    print('Length:', len(ee_mrr_combine_list))

    df = pd.DataFrame(ee_mrr_combine_list, columns=['concept', 'neighbor', 'MRR'])
    df = df.merge(ee_1_df[['concept', 'neighbor', pred_1_rank]], how='left', on=['concept', 'neighbor'])
    df = df.merge(ee_2_df[['concept', 'neighbor', pred_2_rank]], how='left', on=['concept', 'neighbor'])
    df.to_csv(dest, index=None)
    
    
def main():
    args = parse_arguments()

    MRR_combine(**vars(args))

    
if __name__ == "__main__":
    main()