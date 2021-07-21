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

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, load_benchmark

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pred', '--predictions_path', type=str,
                        required=True, help='File with predicted relations')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-rank', '--ranking_by', type=str, default=None,
                        required=False, help='Ranking candidates by which column')
    parser.add_argument('-rev', '--ranking_reverse', action='store_true', help='If set, ranking from high to low')
    args = parser.parse_args()
    return args


def evaluate_EE(predictions_path,
                seed_concepts_path,
                seed_relations_path,
                benchmark_full_path,
                ranking_by,
                ranking_reverse,
                **kwargs):
    '''Format of prediction file: CSV, with column "concept" and "neighbor"(entity), and "{ranking_by}" '''
    preds_df = pd.read_csv(predictions_path)
    
    all_benchmark_instances, _ = load_benchmark(benchmark_full_path, seed_concepts_path, seed_relations_path)
    seed_aligned_concepts = load_seed_aligned_concepts(seed_concepts_path)
    
    mrr_dict = dict()
    recall_100_dict = dict()
    recall_1k_dict = dict()
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

        print(f'Concept: {a_concept} / {u_concept}')
        print(f'seeds: {seed_instances}')
        b_inst_ranks = dict()
        recip_ranks = []
        rec_n_100 = 0
        rec_n_1k = 0
        for _inst in benchmark_instances:
            if _inst in seed_instances:
                b_inst_ranks[_inst] = -1
            elif _inst in pred_instances:
                _rank = pred_instances.index(_inst) + 1
                b_inst_ranks[_inst] = _rank
                recip_ranks.append(1.0 / _rank)
                if _rank <= 100:
                    rec_n_100 += 1
                if _rank <= 1000:
                    rec_n_1k += 1
            else:
                b_inst_ranks[_inst] = float('nan')
                recip_ranks.append(0.0)
                
        mrr = np.mean(recip_ranks) if len(recip_ranks) > 0 else 0.0
        mrr_dict[a_concept] = mrr
        rec_100 = 1.0 * rec_n_100 / len(non_seed_instances)
        recall_100_dict[a_concept] = rec_100
        rec_1k = 1.0 * rec_n_1k / len(non_seed_instances)
        recall_1k_dict[a_concept] = rec_1k
        print(json.dumps(b_inst_ranks, indent=4))
        print('MRR:', mrr)
        print('R@100:', rec_100)
        print('R@1k:', rec_1k)
        print()

    print('--- Summary ---')
    # print(json.dumps(mrr_dict, indent=2))
    print('{:24s}{:^8s}{:^8s}{:^8s}'.format('Concept', 'MRR', 'R@100', 'R@1k'))
    for cc in seed_aligned_concepts['alignedCategoryName'].tolist():
        print('{:24s}{:^8.4f}{:^8.4f}{:^8.4f} '.format(cc, mrr_dict[cc], recall_100_dict[cc], recall_1k_dict[cc]))
    print()
    

def main():
    args = parse_arguments()
    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
    args.benchmark_full_path = os.path.join(args.benchmark_path, 'benchmark_evidence_clean.csv')

    evaluate_EE(**vars(args))

if __name__ == "__main__":
    main()


