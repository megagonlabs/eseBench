from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import os
import torch
import pandas as pd
import json
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, load_benchmark, load_EE_labels

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '-pred', '--predictions_path', type=str,
                        required=True, help='EE predictions file path')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark file path')
    parser.add_argument('-ep', '--entity_props_path', type=str,
                        required=True, help='Entity properties file path')
    parser.add_argument('-b_col', '--benchmark_label_col', type=str, default=None,
                        required=False, help='Benchmark file column for label. If provided, this value > 0 means entity correct. (Example: "Majority") If None (default), assume all entities listed are correct.')
    parser.add_argument('-s', '--seed_concepts_path', type=str,
                        required=True, help='Seed concept-entities file path')
    parser.add_argument('-o', '--result_file_path', type=str, default=None,
                        required=False, help='Output result file path')
    parser.add_argument('-rank', '--ranking_by', type=str, default=None,
                        required=False, help='Ranking candidates by which column')
    parser.add_argument('-rev', '--ranking_reverse', action='store_true', help='If set, ranking from high to low')
    parser.add_argument('-v', '--verbose', action='store_true', help='If set, print more info')
    args = parser.parse_args()
    return args


def evaluate_EE_props(predictions_path,
                seed_concepts_path,
                entity_props_path,
                benchmark_path,
                benchmark_label_col,
                ranking_by,
                ranking_reverse,
                result_file_path,
                verbose,
                **kwargs):
    '''
    Format of prediction file: CSV, with column "concept" and "neighbor"(entity), and "{ranking_by}"
    Format of output file: CSV, header = concept, property, category (bool), hit, total, recall
        concept == 'all' means stats on all entities (regardless of concept)
        concept == 'avg' means stats averaged across concepts
        property == 'all' means stats on all entities under the concept (regardless of properties)
        
    '''
    
    K_list = [5, 10, 20, 50, 100, 200, 300]
    props = ['multifaceted', 'non_named']  # TODO: 'vague' 

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    preds_df = pd.read_csv(predictions_path)

    props_df = pd.read_csv(entity_props_path)
    ent2props = dict()   # Dict[Tuple:(cc, e), Dict[str:prop, bool]]
    for d in props_df.to_dict('record'):
        cc = d['concept']
        e = d['neighbor'].lower()
        _prop_d = dict()
        for p in props:
            _prop_d[p] = d[p]
        ent2props[(cc, e)] = _prop_d

    #     all_benchmark_instances = load_EE_labels(ee_labels_path)
    all_benchmark_instances = load_EE_labels(benchmark_path, label_col=benchmark_label_col)

    seed_aligned_concepts = load_seed_aligned_concepts(seed_concepts_path)

    ## Dict[str:concept or 'all',
    ##    Dict[Tuple:(prop,category) or ('all',None), Counter['hit', 'total', 'recall']]]
    recall_at_k_dicts = dict()
    recall_at_k_dicts['all'] = defaultdict(Counter)

    for i, d in seed_aligned_concepts.iterrows():
        a_concept = d["alignedCategoryName"]
        u_concept = d["unalignedCategoryName"]
        seed_instances = d["seedInstances"]
        seed_instances = [e.lower() for e in seed_instances]
        ## It is possible that benchmark doesn't include all concepts (e.g. test-in-concept/domain)
        if a_concept not in all_benchmark_instances:
            continue

    #         concept_knn_instances = concept_knn[concept_knn["concept"] == a_concept]["neighbor"].to_list()
    #         pred_instances = preds_df[preds_df["concept"] == a_concept]["neighbor"].to_list()
        pred_rows = preds_df[preds_df["concept"] == a_concept].to_dict('records')
        if ranking_by is not None:
            assert ranking_by in preds_df.columns, f'{ranking_by} not in {preds_df.columns}'
            random.seed(127)
            random.shuffle(pred_rows)
            pred_rows.sort(key=lambda r: r[ranking_by], reverse=ranking_reverse)

        pred_instances = []
        ## @YS: dedup 
        for r in pred_rows:
            e = r['neighbor'].lower()
            if e not in pred_instances:
                pred_instances.append(e)
        ## @YS: not including seeds for evaluation!
        pred_instances = [_e for _e in pred_instances if _e not in seed_instances]

        benchmark_instances = list(set([e.lower() for e in all_benchmark_instances[a_concept]]))
        non_seed_instances = [e for e in benchmark_instances if e not in seed_instances]
        gold_k = len(non_seed_instances)

        vprint(f'Concept: {a_concept} / {u_concept}')
        vprint(f'seeds: {seed_instances}')
        vprint(f'preds (top-20): {pred_instances[:20]}')

        ## Dict[Tuple/str:(prop,category) or 'all', Counter['hit', 'total']]
        cc_recall_dict = defaultdict(Counter)

        for i, k in enumerate(K_list + [gold_k]):
            # k_str: distinguish fixed K and gold_k 
            k_str = str(k) if i < len(K_list) else 'K'

            selected_instances = set(pred_instances[:k])
            for _inst in non_seed_instances:
                hit = _inst in selected_instances
                for prop, cate in list(ent2props[(a_concept, _inst)].items()) + [('all', None)]:
                    recall_at_k_dicts['all'][(prop, cate)][f'hit@{k_str}'] += hit
                    recall_at_k_dicts['all'][(prop, cate)][f'total@{k_str}'] += 1
                    cc_recall_dict[(prop, cate)][f'hit@{k_str}'] += hit
                    cc_recall_dict[(prop, cate)][f'total@{k_str}'] += 1

        assert cc_recall_dict[('all', None)]['total@K'] == gold_k, \
            (cc_recall_dict[('all', None)], gold_k)

        for key, cnter in cc_recall_dict.items():
            for k in K_list + ['K']:
                if cnter[f'total@{k}'] == 0:
                    rec = float('nan')
                else:
                    rec = 1.0 * cnter[f'hit@{k}'] / cnter[f'total@{k}']
                cnter[f'recall@{k}'] = rec
            vprint(key, cnter)

        recall_at_k_dicts[a_concept] = cc_recall_dict

    vprint(f'Concept: All')
    for key, cnter in recall_at_k_dicts['all'].items():
        for k in K_list + ['K']:
            if cnter[f'total@{k}'] == 0:
                rec = float('nan')
            else:
                rec = 1.0 * cnter[f'hit@{k}'] / cnter[f'total@{k}']
            cnter[f'recall@{k}'] = rec
        vprint(key, cnter)

    all_prop_keys = [('all', None)] + [(prop, cate) for prop in props for cate in [True, False]]
    vprint(f'Concept: Avg across concepts')
    recall_at_k_dicts['avg'] = defaultdict(Counter)
    for prop, cate in all_prop_keys:
        for stat in [f'{s}@{k}' for s in ['total', 'hit', 'recall'] for k in K_list + ['K']]:
                all_cc_stats = [recall_at_k_dicts[cc][(prop, cate)][stat] for cc in recall_at_k_dicts.keys() if cc != 'all']
                avg_cc_stat = np.mean(all_cc_stats)
                recall_at_k_dicts['avg'][(prop, cate)][stat] = avg_cc_stat
    vprint(recall_at_k_dicts['avg'])

    out_records = []   # List[Dict['property', 'category', 'R@K', 'R@5|10|...']]
    if result_file_path is not None:
        # Dump to csv
        for cc, cc_recall_dict in recall_at_k_dicts.items():
            for prop, cate in all_prop_keys:
                    _d = cc_recall_dict[(prop, cate)]
                    rec_d = {
                        'concept': cc,
                        'property': prop,
                        'category': cate
                    }

                    for stat in [f'{s}@{k}' for k in K_list + ['K'] for s in ['total', 'hit', 'recall']]:
                        rec_d[stat] = _d[stat]

                    out_records.append(rec_d)

        pd.DataFrame(out_records).to_csv(result_file_path, index=None)

    return recall_at_k_dicts, out_records


def main():
    args = parse_arguments()
#     args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
#     args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
#     args.seed_relations_path = None
#     args.benchmark_full_path = os.path.join(args.benchmark_path, 'benchmark_evidence_clean.csv')
#     args.ee_labels_path = os.path.join(args.benchmark_path, 'ee-labels-2.csv')

    evaluate_EE_props(**vars(args))

if __name__ == "__main__":
    main()


