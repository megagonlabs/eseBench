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

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, load_benchmark, load_EE_labels

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', '--predictions_path', type=str,
                        required=True, help='File with predicted relations')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-o', '--result_file_path', type=str, default=None,
                        required=False, help='Output result file path')
    parser.add_argument('-rank', '--ranking_by', type=str, default=None,
                        required=False, help='Ranking candidates by which column')
    parser.add_argument('-rev', '--ranking_reverse', action='store_true', help='If set, ranking from high to low')
    parser.add_argument('-v', '--verbose', action='store_true', help='If set, print more info')
    args = parser.parse_args()
    return args


def evaluate_EE(predictions_path,
                seed_concepts_path,
                seed_relations_path,
                benchmark_full_path,
                ee_labels_path,
                ranking_by,
                ranking_reverse,
                result_file_path,
                verbose,
                **kwargs):
    '''Format of prediction file: CSV, with column "concept" and "neighbor"(entity), and "{ranking_by}" '''
    
    K_list = [5, 10, 20, 50, 100, 200, 300]
    
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    preds_df = pd.read_csv(predictions_path)
    
#     all_benchmark_instances, _ = load_benchmark(benchmark_full_path, seed_concepts_path, seed_relations_path)
    all_benchmark_instances = load_EE_labels(ee_labels_path)
    seed_aligned_concepts = load_seed_aligned_concepts(seed_concepts_path)
    
    mrr_dict = dict()
    max_k_dict = dict()
    recall_at_k_dicts = dict()  # Dict[cc, Dict[k, recall]]
    prec_at_k_dicts = dict()    # Dict[cc, Dict[k, recall]]
    for i, d in seed_aligned_concepts.iterrows():
        a_concept = d["alignedCategoryName"]
        u_concept = d["unalignedCategoryName"]
        seed_instances = d["seedInstances"]

#         concept_knn_instances = concept_knn[concept_knn["concept"] == a_concept]["neighbor"].to_list()
#         pred_instances = preds_df[preds_df["concept"] == a_concept]["neighbor"].to_list()
        pred_rows = preds_df[preds_df["concept"] == a_concept].to_dict('records')
        if ranking_by is not None:
            assert ranking_by in preds_df.columns, f'{ranking_by} not in {preds_df.columns}'
            pred_rows.sort(key=lambda r: r[ranking_by], reverse=ranking_reverse)
        pred_instances = [r['neighbor'] for r in pred_rows]

#         _b_head_instances = benchmark[benchmark["n_head_category"] == a_concept]["n_head"].to_list()
#         _b_tail_instances = benchmark[benchmark["n_tail_category"] == a_concept]["n_tail"].to_list()
#         benchmark_instances = list(set(_b_head_instances + _b_tail_instances))
        benchmark_instances = all_benchmark_instances[a_concept]
        non_seed_instances = [e for e in benchmark_instances if e not in seed_instances]

        vprint(f'Concept: {a_concept} / {u_concept}')
        vprint(f'seeds: {seed_instances}')
        b_inst_ranks = dict()
        recip_ranks = []
        hit_n_dict = dict([(k, 0) for k in K_list])
        for _inst in benchmark_instances:
            if _inst in seed_instances:
                b_inst_ranks[_inst] = -1
            elif _inst in pred_instances:
                _rank = pred_instances.index(_inst) + 1
                b_inst_ranks[_inst] = _rank
                recip_ranks.append(1.0 / _rank)
                for k in reversed(K_list):
                    if _rank <= k:
                        hit_n_dict[k] += 1
                    else:
                        break
            else:
                b_inst_ranks[_inst] = float('nan')
                recip_ranks.append(0.0)
                
        mrr = np.mean(recip_ranks) if len(recip_ranks) > 0 else 0.0
        mrr_dict[a_concept] = mrr
#         r_100 = 1.0 * hit_n_100 / len(non_seed_instances)
#         p_100 = 1.0 * hit_n_100 / min(100, len(pred_instances))
#         recall_100_dict[a_concept] = r_100
#         prec_100_dict[a_concept] = p_100
#         r_1k = 1.0 * hit_n_1k / len(non_seed_instances)
#         p_1k = 1.0 * hit_n_1k / min(1000, len(pred_instances))
#         recall_1k_dict[a_concept] = r_1k
#         prec_1k_dict[a_concept] = p_1k
        _recall_d = dict()
        _prec_d = dict()
        for k in K_list:
            _r = 1.0 * hit_n_dict[k] / len(non_seed_instances)
            _p = 1.0 * hit_n_dict[k] / min(k, len(pred_instances))
            _recall_d[k] = _r
            _prec_d[k] = _p
        recall_at_k_dicts[a_concept] = _recall_d
        prec_at_k_dicts[a_concept] = _prec_d
        max_k_dict[a_concept] = len(pred_instances)
        
        vprint(json.dumps(b_inst_ranks, indent=4))
        vprint('MRR:', mrr)
        vprint('Max K:', len(pred_instances))
#         vprint('P@100:', p_100)
#         vprint('R@100:', r_100)
#         vprint('P@1k:', p_1k)
#         vprint('R@1k:', r_1k)
        for k in K_list:
            vprint(f'P@{k}:', _prec_d[k])
            vprint(f'R@{k}:', _recall_d[k])
        vprint()

    print('--- Summary ---')
    # print(json.dumps(mrr_dict, indent=2))
    print('{:24s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}'.format('Concept', 'Max K', 'MRR', 'P@20', 'R@20', 'P@100', 'R@100'))
    for cc in seed_aligned_concepts['alignedCategoryName'].tolist():
        print('{:24s}{:^8d}{:^8.4f}{:^8.4f}{:^8.4f}{:^8.4f}{:^8.4f}'.format(cc, max_k_dict[cc], mrr_dict[cc], prec_at_k_dicts[cc][20], recall_at_k_dicts[cc][20], prec_at_k_dicts[cc][100], recall_at_k_dicts[cc][100]))
    print()
    
    if result_file_path is not None:
        # Dump to csv
        _cc_records = []
        for cc in seed_aligned_concepts['alignedCategoryName'].tolist():
            _d = {
                'concept': cc,
                'max_k': max_k_dict[cc],
                'MRR': mrr_dict[cc],
            }
            for k in K_list:
                _p = prec_at_k_dicts[cc][k]
                _r = recall_at_k_dicts[cc][k]
                _f1 = 2 * _p * _r / (_p + _r + 1e-9)
                _d[f'P@{k}'] = _p
                _d[f'R@{k}'] = _r
                _d[f'F1@{k}'] = _f1
            _cc_records.append(_d)
        pd.DataFrame(_cc_records).to_csv(result_file_path, index=None)
    
    return prec_at_k_dicts, recall_at_k_dicts

def main():
    args = parse_arguments()
    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
    args.benchmark_full_path = os.path.join(args.benchmark_path, 'benchmark_evidence_clean.csv')
    args.ee_labels_path = os.path.join(args.benchmark_path, 'ee-labels.csv')

    evaluate_EE(**vars(args))

if __name__ == "__main__":
    main()


