from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import os
import torch
import pandas as pd
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', '--prediction_path', type=str,
                        required=True, help='File with predicted relations')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-r', '--relation', type=str, required=False,
                        default=None, help='The relation to evaluate for (None means all relations)')
    args = parser.parse_args()
    return args


# def load_seed_aligned_concepts(path):
#     df = pd.read_csv(path)
#     df = df[df["generalizations"] != "x"]
#     df["seedInstances"] = df["seedInstances"].map(lambda s : eval(str(s)))
#     return df

# def load_seed_aligned_relations(path):
#     df = pd.read_csv(path)
#     df = df[df["range"] != "x"]
#     return df


def load_benchmark(benchmark_full_path,
                   seed_concepts_path,
                   seed_relations_path):
    benchmark = pd.read_csv(benchmark_full_path)
    concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    relations_df = load_seed_aligned_relations(seed_relations_path)
    
    concepts_dict = dict(zip(concepts_df['alignedCategoryName'].tolist(), concepts_df.to_dict('records')))
    relations_dict = dict(zip(relations_df['alignedRelationName'].tolist(), relations_df.to_dict('records')))
    
    # Dict[str(_type), Set[str(_e)]]
    all_concepts = defaultdict(set)
    # Dict[str(_r), Set[Tuple(_h, _r, _t)]]
    all_rel_tuples = defaultdict(set)
    
    for i, row in benchmark.iterrows():
        if row['type'] != 'fact':
            continue
        
        _r = row['relation_name']
        _h_type = row['n_head_category']
        _t_type = row['n_tail_category']
        
        if _r not in relations_dict:
            continue
        _relation_row = relations_dict[_r]
        if _relation_row['domain'] != _h_type or _relation_row['range'] != _t_type:
            continue
        
        row_n_head = str(row['n_head']).lower()
        row_n_tail = str(row['n_tail']).lower()
        
        if row_n_head == 'company':
            evidence_sents = eval(str(row['sentences']))
            head_instances = eval(str(row['Evidence']))
            assert len(evidence_sents) == len(head_instances), f'Line {i} length mismatch'

            for inst in head_instances:
                if len(inst) > 0:
                    all_concepts[_h_type].add(inst)
                    all_concepts[_t_type].add(row_n_tail)
                    all_rel_tuples[_r].add(
                        (inst.lower(), _r, row_n_tail)
                    )
        else:
            # treat n_head directly as instance 
            all_concepts[_h_type].add(row_n_head)
            all_concepts[_t_type].add(row_n_tail)
            all_rel_tuples[_r].add(
                (row_n_head, _r, row_n_tail)
            )
        
    return all_concepts, all_rel_tuples



def evaluate_RE(prediction_path,
                seed_concepts_path,
                seed_relations_path,
                benchmark_full_path,
                relation=None,
                **kwargs):
    '''
    Format of input file:
        CSV, with columns 'head', 'relation' and 'tail' (can include others)
        All these cell values should be the aligned version and lower-case
    '''
    
#     benchmark_rels_list = load_benchmark_relations(benchmark_full_path, seed_relations_path)
#     benchmark_rels = set(benchmark_rels_list)
    all_bench_concepts, all_bench_rel_tuples = load_benchmark(benchmark_full_path, seed_concepts_path, seed_relations_path)
    if relation is not None:
        benchmark_rels = all_bench_rel_tuples[relation]
    else:
        benchmark_rels = set()
        for _rel_name, _rel_tuples in all_bench_rel_tuples.items():
            benchmark_rels.update(_rel_tuples)

    pred_rels_df = pd.read_csv(prediction_path)
    pred_rels = pred_rels_df[['head', 'relation', 'tail']].to_records(index=False).tolist()
    pred_rels = set(pred_rels)
    
    intersect_rels = benchmark_rels & pred_rels
    
    n_bench = len(benchmark_rels)
    n_pred = len(pred_rels)
    n_intersect = len(intersect_rels)
    P = 1.0 * n_intersect / n_pred
    R = 1.0 * n_intersect / n_bench
    if n_intersect == 0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    
    print('--- RE Results ---')
    print(f'Benchmark relations: {n_bench}')
    print(f'Predicted relations: {n_pred}')
    print(f'Intersection: {n_intersect}')
    print(f'P = {P:.4f}, R = {R:.4f}, F1 = {F1:.4f}')
    print()
    print('Intersection:')
    for _rel in sorted(list(intersect_rels)):
        print(_rel)
    print()
    

def main():
    args = parse_arguments()
    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
    args.benchmark_full_path = os.path.join(args.benchmark_path, 'benchmark_evidence_clean.csv')

    evaluate_RE(**vars(args))

if __name__ == "__main__":
    main()


