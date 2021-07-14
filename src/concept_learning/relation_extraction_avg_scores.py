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

from utils import load_seed_aligned_concepts, load_seed_aligned_relations, load_embeddings, LMProbe

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
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')
    parser.add_argument('-topk', '--topk', type=int, default=300,
                        help='The number of entities to keep for each base head/tail')

    args = parser.parse_args()
    return args


def direct_probing_RE_v3(seed_concepts_path,
                         seed_relations_path,
                         emb_path,
                         concept_knn_path,
                         templates_path,
                         relation,
                         lm_probe=None,
                         embedding_dim=768,
                         scores_agg_func=None,
                         topk=300,
                         dest=None,
                         **kwargs):
    '''
    For each head / tail, rank candidate tails / heads by overall scores. 
    Current (default) overall score: 0.1 * ht_sim + 10 * concept_sim + 0.1 * log(lm_prob)
    '''
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_relations_df = pd.read_csv(seed_relations_path)
    relation_row = seed_relations_df[seed_relations_df['alignedRelationName'] == relation].iloc[0]
    entity_embeddings = load_embeddings(emb_path, embedding_dim)
    entity_emb_dict = dict(zip(entity_embeddings['entity'].tolist(),
                               entity_embeddings['embedding'].tolist()))
    concept_knn_results = pd.read_csv(concept_knn_path)
    
    with open(templates_path, 'r') as f:
        all_templates = json.load(f)
    templates = all_templates[relation]
    templates = templates['positive'] + templates['negative']

    if lm_probe is None:
        lm_probe = LMProbe()
    if scores_agg_func is None:
        scores_agg_func = lambda ht_sim, concept_sim, lm_prob : 0.1 * ht_sim + 10 * concept_sim + 0.1 * np.log10(lm_prob)
    
    head_type = relation_row['domain']
    tail_type = relation_row['range']
#     head_type = "company"
#     tail_type = "dress_code"
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
    cand_heads_dict = dict(zip(cand_heads_df['neighbor'].tolist(), cand_heads_df['sim'].tolist()))
    cand_tails_dict = dict(zip(cand_tails_df['neighbor'].tolist(), cand_tails_df['sim'].tolist()))
    for h in seed_heads:
        assert h not in cand_heads_dict
        cand_heads_dict[h] = 1.0
    for t in seed_tails:
        assert t not in cand_tails_dict
        cand_tails_dict[t] = 1.0
    
#     print(cand_heads_dict)
#     print(cand_tails_dict)
    
    all_extraction_results = []
    
    # head -> tail 
    for seed_head in seed_heads:
        print(f'seed_head: {seed_head}')
        extraction_results = []

        ## For each tail, extract concept sim, head sim, lm score, combine and report
        
        cand_bins = {1: [], 2: []} ## TODO: allow higher grams; switch to GPT-2 for fair probs 
        for c in cand_tails_dict.keys():
            c_tokenized = lm_probe.tokenizer.tokenize(c)
            if len(c_tokenized) in [1, 2]:
                cand_bins[len(c_tokenized)].append(c_tokenized)
        
        cand_scores_per_template = []
        for template in templates:
            _unigram_template = '[CLS] ' + template.format(seed_head, '[MASK]') + '[SEP]'
            _bigram_template = '[CLS] ' + template.format(seed_head, '[MASK] [MASK]') + '[SEP]'

            _cand_scores_1 = lm_probe.score_candidates(_unigram_template, cand_bins[1])
            _cand_scores_2 = lm_probe.score_candidates(_bigram_template, cand_bins[2])
            _cand_scores = sorted(_cand_scores_1 + _cand_scores_2, key=lambda d : d["cand"])
            # List[Dict["cand", "score"]]
            cand_scores_per_template.append(_cand_scores)
    
        cand_scores = []  # List[Dict["cand", "score"]], for each "cand" the average score 
        for _cand_score_lst in zip(*cand_scores_per_template):
            # _cand_score_lst: List[Dict["cand", "score"]], for the same "cand" and different template 
            _cand = _cand_score_lst[0]["cand"]
            assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
            _score = np.mean([d["score"] for d in _cand_score_lst])
            cand_scores.append({"cand": _cand, "score": _score})
#         cand_scores.sort(key = lambda d : d["score"], reverse=True)

        for d in cand_scores:
            e_tail = ' '.join(d["cand"]).replace(' ##', '')
            if e_tail not in cand_tails_dict:
                continue

            lm_score = d["score"]
            try:
                ht_sim_score = 1 - cosine(entity_emb_dict[seed_head], entity_emb_dict[e_tail])
            except KeyError:
                print(f'** embedding of {seed_head}: {(seed_head in entity_emb_dict)}')
                print(f'** embedding of {e_tail}: {(e_tail in entity_emb_dict)}')
                ht_sim_score = float("nan")
            concept_sim_score = cand_tails_dict[e_tail]
            overall_score = scores_agg_func(ht_sim_score, concept_sim_score, lm_score)

            extraction_results.append({'head': seed_head, 'relation': relation, 'tail': e_tail,
                                       'base': 'HEAD',
                                       'ht_sim_score': ht_sim_score,
                                       'concept_sim_score': concept_sim_score,
                                       'lm_score': lm_score,
                                       'overall_score': overall_score})
        
        extraction_results.sort(key=lambda d : d['overall_score'], reverse=True)
        all_extraction_results.extend(extraction_results[:topk])
        
    # tail -> head 
    for seed_tail in seed_tails:
        print(f'seed_tail: {seed_tail}')
        extraction_results = []
        
        ## For each tail, extract concept sim, head sim, lm score, combine and report
        
        cand_bins = {1: [], 2: []}
        for c in cand_heads_dict.keys():
            c_tokenized = lm_probe.tokenizer.tokenize(c)
            if len(c_tokenized) in [1, 2]:
                cand_bins[len(c_tokenized)].append(c_tokenized)
        
        cand_scores_per_template = []
        for template in templates:
            _unigram_template = '[CLS] ' + template.format('[MASK]', seed_tail) + '[SEP]'
            _bigram_template = '[CLS] ' + template.format('[MASK] [MASK]', seed_tail) + '[SEP]'

            _cand_scores_1 = lm_probe.score_candidates(_unigram_template, cand_bins[1])
            _cand_scores_2 = lm_probe.score_candidates(_bigram_template, cand_bins[2])
            _cand_scores = sorted(_cand_scores_1 + _cand_scores_2, key=lambda d : d["cand"])
            # List[Dict["cand", "score"]]
            cand_scores_per_template.append(_cand_scores)
    
        cand_scores = []  # List[Dict["cand", "score"]], for each "cand" the average score 
        for _cand_score_lst in zip(*cand_scores_per_template):
            # _cand_score_lst: List[Dict["cand", "score"]], for the same "cand" and different template 
            _cand = _cand_score_lst[0]["cand"]
            assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
            _score = np.mean([d["score"] for d in _cand_score_lst])
            cand_scores.append({"cand": _cand, "score": _score})
#         cand_scores.sort(key = lambda d : d["score"], reverse=True)

        for d in cand_scores[:topk]:
            e_head = ' '.join(d["cand"]).replace(' ##', '')
            if e_head not in cand_heads_dict:
                continue
                
            lm_score = d["score"]
            try:
                ht_sim_score = 1 - cosine(entity_emb_dict[e_head], entity_emb_dict[seed_tail])
            except KeyError:
                print(f'** embedding of {e_head}: {(e_head in entity_emb_dict)}')
                print(f'** embedding of {seed_tail}: {(seed_tail in entity_emb_dict)}')
                ht_sim_score = float("nan")
            concept_sim_score = cand_heads_dict[e_head]
            overall_score = scores_agg_func(ht_sim_score, concept_sim_score, lm_score)
        
            extraction_results.append({'head': e_head, 'relation': relation, 'tail': seed_tail,
                                       'base': 'TAIL',
                                       'ht_sim_score': ht_sim_score,
                                       'concept_sim_score': concept_sim_score,
                                       'lm_score': lm_score,
                                       'overall_score': overall_score})
        
        extraction_results.sort(key=lambda d : d['overall_score'], reverse=True)
        all_extraction_results.extend(extraction_results[:topk])
        
    results_df = pd.DataFrame(all_extraction_results)
    if dest is not None:
        results_df.to_csv(dest, index=None)
    return results_df


def direct_probing_RE_v4(seed_concepts_path,
                         seed_relations_path,
                         emb_path,
                         concept_knn_path,
                         templates_path,
                         relation,
                         lm_probe=None,
                         embedding_dim=768,
                         scores_agg_func=None,
                         topk=300,
                         dest=None,
                         **kwargs):
    '''
    For each head / tail, rank candidate tails / heads by overall scores. 
    (v4: Not limited to base -> new; can be new -> new; however, only head->tail, no tail->head)
    Current (default) overall score: ht_sim + h_sim + t_sim + log(lm_prob)
    '''
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_relations_df = pd.read_csv(seed_relations_path)
    relation_row = seed_relations_df[seed_relations_df['alignedRelationName'] == relation].iloc[0]
    entity_embeddings = load_embeddings(emb_path, embedding_dim)
    entity_emb_dict = dict(zip(entity_embeddings['entity'].tolist(),
                               entity_embeddings['embedding'].tolist()))
    concept_knn_results = pd.read_csv(concept_knn_path)
    
    with open(templates_path, 'r') as f:
        all_templates = json.load(f)
    templates = all_templates[relation]
    templates = templates['positive'] + templates['negative']

    if lm_probe is None:
        lm_probe = LMProbe()
    if scores_agg_func is None:
        scores_agg_func = lambda ht_sim, h_sim, t_sim, lm_prob : ht_sim + h_sim + t_sim + np.log10(lm_prob)
    
    head_type = relation_row['domain']
    tail_type = relation_row['range']
#     head_type = "company"
#     tail_type = "dress_code"
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
    cand_heads_dict = dict(zip(cand_heads_df['neighbor'].tolist(), cand_heads_df['sim'].tolist()))
    cand_tails_dict = dict(zip(cand_tails_df['neighbor'].tolist(), cand_tails_df['sim'].tolist()))
    for h in seed_heads:
        assert h not in cand_heads_dict
        cand_heads_dict[h] = 1.0
    for t in seed_tails:
        assert t not in cand_tails_dict
        cand_tails_dict[t] = 1.0
        
    
    all_extraction_results = []
    
    for c_head in tqdm(cand_heads_dict.keys(), total=len(cand_heads_dict)):
        c_head_tokenized = lm_probe.tokenizer.tokenize(c_head)
        if len(c_head_tokenized) > 2:
            continue

        extraction_results = []

        ## For each tail, extract concept sim, head sim, lm score, combine and report
        
        cand_bins = {1: [], 2: []} ## TODO: allow higher grams; switch to GPT-2 for fair probs 
        for c_tail in cand_tails_dict.keys():
            if c_tail == c_head:
                continue
            c_tail_tokenized = lm_probe.tokenizer.tokenize(c_tail)
            if len(c_tail_tokenized) in [1, 2]:
                cand_bins[len(c_tail_tokenized)].append(c_tail_tokenized)
        
        cand_scores_per_template = []
        for template in templates:
            _unigram_template = '[CLS] ' + template.format(c_head, '[MASK]') + '[SEP]'
            _bigram_template = '[CLS] ' + template.format(c_head, '[MASK] [MASK]') + '[SEP]'

            _cand_scores_1 = lm_probe.score_candidates(_unigram_template, cand_bins[1])
            _cand_scores_2 = lm_probe.score_candidates(_bigram_template, cand_bins[2])
            _cand_scores = sorted(_cand_scores_1 + _cand_scores_2, key=lambda d : d["cand"])
            # List[Dict["cand", "score"]]
            cand_scores_per_template.append(_cand_scores)
    
        cand_scores = []  # List[Dict["cand", "score"]], for each "cand" the average score 
        for _cand_score_lst in zip(*cand_scores_per_template):
            # _cand_score_lst: List[Dict["cand", "score"]], for the same "cand" and different template 
            _cand = _cand_score_lst[0]["cand"]
            assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
            _score = np.mean([d["score"] for d in _cand_score_lst])
            cand_scores.append({"cand": _cand, "score": _score})
#         cand_scores.sort(key = lambda d : d["score"], reverse=True)

        for d in cand_scores:
            e_tail = ' '.join(d["cand"]).replace(' ##', '')
            if e_tail not in cand_tails_dict:
                continue

            lm_score = d["score"]
            try:
                ht_sim_score = 1 - cosine(entity_emb_dict[c_head], entity_emb_dict[e_tail])
            except KeyError:
                print(f'** embedding of {c_head}: {(c_head in entity_emb_dict)}')
                print(f'** embedding of {e_tail}: {(e_tail in entity_emb_dict)}')
                ht_sim_score = float("nan")
            head_sim_score = cand_heads_dict[c_head]
            tail_sim_score = cand_tails_dict[e_tail]
            overall_score = scores_agg_func(ht_sim_score, head_sim_score, tail_sim_score, lm_score)

            extraction_results.append({'head': c_head, 'relation': relation, 'tail': e_tail,
                                       'ht_sim_score': ht_sim_score,
                                       'head_sim_score': head_sim_score,
                                       'tail_sim_score': tail_sim_score,
                                       'lm_score': lm_score,
                                       'overall_score': overall_score})
        
        # extraction_results.sort(key=lambda d : d['overall_score'], reverse=True)
        all_extraction_results.extend(extraction_results[:topk])

    all_extraction_results.sort(key=lambda d : d['overall_score'], reverse=True)
    all_extraction_results = all_extraction_results[:topk]
        
    results_df = pd.DataFrame(all_extraction_results)
    if dest is not None:
        results_df.to_csv(dest, index=None)
    return results_df



    
def main():
    args = parse_arguments()
    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations_nodup.csv')
    args.emb_path = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
    args.templates_path = 'templates_manual.json'
    
#     args.dest = os.path.join(args.dataset_path, 'rel_extraction.csv')

    direct_probing_RE_v4(**vars(args))
    
    
if __name__ == "__main__":
    main()