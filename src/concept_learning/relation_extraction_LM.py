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
from lm_probes import LMProbe, LMProbe_GPT2, LMProbe_Joint, LMProbe_PMI, LMProbe_PMI_greedy

from roberta_ses.interface import Roberta_SES_Entailment

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-ee', '--EE_path', type=str,
                        required=True, help='Path to EE prediction file')
    parser.add_argument('-lm', '--lm_probe_type', type=str,
                        required=True, help='The type of lm_probe to use')
    parser.add_argument('-ex', '--template_extra_type', type=str, default=None,
                        required=False, help='The type of extra tokens to add into templates')
    parser.add_argument('-r', '--relation', type=str, required=True,
                        help='The relation to extract for')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to output file')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')
    parser.add_argument('-topk', '--topk', type=int, default=3000,
                        help='The number of entities to keep for each base head/tail')

    args = parser.parse_args()
    return args


def RE_LMProbe(seed_concepts_path,
               seed_relations_path,
               emb_path,
               EE_path,
               templates_path,
               relation,
               lm_probe_type,
               template_extra_type=None,
               embedding_dim=768,
               scores_agg_func=None,
               tmpl_agg_func=None,
               max_n_grams=5,
               topk=300,
               dest=None,
               **kwargs):
    '''
    For each head / tail, rank candidate tails / heads by overall scores. 
    (v4: Not limited to base -> new; can be new -> new; however, only head->tail, no tail->head)
    Current (default) overall score: h_sim + t_sim + lm_score (log_prob)
    '''
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_relations_df = pd.read_csv(seed_relations_path)
    relation_row = seed_relations_df[seed_relations_df['alignedRelationName'] == relation].iloc[0]
    entity_embeddings = load_embeddings(emb_path, embedding_dim)
    entity_emb_dict = dict(zip(entity_embeddings['entity'].tolist(),
                               entity_embeddings['embedding'].tolist()))
    EE_results = pd.read_csv(EE_path)
    
    with open(templates_path, 'r') as f:
        all_templates = json.load(f)
    templates = all_templates[relation]
    templates = templates['positive'] + templates['negative']

    if lm_probe_type == 'bert':
        lm_probe = LMProbe()
    elif lm_probe_type == 'gpt2':
        lm_probe = LMProbe_GPT2()
    elif lm_probe_type == 'joint':
        lm_probe = LMProbe_Joint()
    elif lm_probe_type == 'pmi':
        lm_probe = LMProbe_PMI()
    elif lm_probe_type == 'pmi_greedy':
        lm_probe = LMProbe_PMI_greedy()
    else:
        raise NotImplementedError(f"lm_probe_type = {lm_probe_type}")

    if scores_agg_func is None:
        scores_agg_func = lambda h_sim, t_sim, lm_score : h_sim + t_sim + lm_score
    if tmpl_agg_func is None:
        tmpl_agg_func = max
    
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
    
    
    if template_extra_type is None:
        probe_prompts = templates
    elif template_extra_type == 'seeds':
        _seeds = seed_tails[:2]
        probe_prompts = []
        for _tmpl in templates:
            probe_prompts.append(_tmpl.format('{0}', f'{_seeds[0]}, {{1}} and {_seeds[1]}'))
            probe_prompts.append(_tmpl.format('{0}', f'{_seeds[0]}, {{1}} or {_seeds[1]}'))
    

    # Candidate heads / tails from concept knn 
    cand_heads_df = EE_results[EE_results['concept'] == head_type]
    cand_tails_df = EE_results[EE_results['concept'] == tail_type]
    cand_heads_dict = dict(zip(cand_heads_df['neighbor'].tolist(), cand_heads_df['sim'].tolist()))
    cand_tails_dict = dict(zip(cand_tails_df['neighbor'].tolist(), cand_tails_df['sim'].tolist()))
    for h in seed_heads:
        assert h not in cand_heads_dict
        cand_heads_dict[h] = 1.0
    for t in seed_tails:
        assert t not in cand_tails_dict
        cand_tails_dict[t] = 1.0
    cand_heads = list(cand_heads_dict.keys())
    cand_tails = list(cand_tails_dict.keys())
        
    all_extraction_results = []
    
    for c_head in tqdm(cand_heads_dict.keys(), total=len(cand_heads_dict)):
        # c_head_tokenized = lm_probe.tokenizer.tokenize(c_head)

        extraction_results = []

        ## For each tail, extract concept sim, head sim, lm score, combine and report
        
        cand_scores = []  # List[Dict["cand", "score"]], for each "cand" the average score 
        
        if lm_probe_type in ['bert', 'gpt2', 'joint']:
            # Score: log_prob
            cand_scores_per_template = []
            for template in probe_prompts:
                # template: {0} = head, {1} = tail
                _input_txt = template.format(c_head, '[MASK]')
                _cand_scores = lm_probe.score_candidates(_input_txt, cand_tails)
                _cand_scores.sort(key=lambda d : d["cand"])
                # List[Dict["cand", "score"]]
                cand_scores_per_template.append(_cand_scores)

            for _cand_score_lst in zip(*cand_scores_per_template):
                # _cand_score_lst: List[Dict["cand", "score"]], for the same "cand" and different template 
                _cand = _cand_score_lst[0]["cand"]
                assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
                _score = tmpl_agg_func([d["score"] for d in _cand_score_lst])
                _score = np.log(_score)
                cand_scores.append({"cand": _cand, "score": _score})
        elif lm_probe_type in ['pmi', 'pmi_greedy']:
            # Score: PMI 
            
            for tail in cand_tails:
                _scores = []
                for template in templates:
                    # template: {0} = head, {1} = tail
                    _input_txt = template.format('[HEAD]', '[TAIL]')
                    _score = lm_probe.score_candidate_pair(_input_txt, head=c_head, tail=tail)
                    _scores.append(_score)

                _score = tmpl_agg_func(_scores)
                cand_scores.append({"cand": tail, "score": _score})
            

        for d in cand_scores:
            e_tail = d["cand"]
            if e_tail not in cand_tails_dict:
                continue
            if e_tail == c_head:
                continue

            lm_score = d["score"]
#             try:
#                 ht_sim_score = 1 - cosine(entity_emb_dict[c_head], entity_emb_dict[e_tail])
#             except KeyError:
#                 print(f'** embedding of {c_head}: {(c_head in entity_emb_dict)}')
#                 print(f'** embedding of {e_tail}: {(e_tail in entity_emb_dict)}')
#                 ht_sim_score = float("nan")
            head_sim_score = cand_heads_dict[c_head]
            tail_sim_score = cand_tails_dict[e_tail]
            overall_score = scores_agg_func(head_sim_score, tail_sim_score, lm_score)

            extraction_results.append({'head': c_head, 'relation': relation, 'tail': e_tail,
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

    RE_LMProbe(**vars(args))
    
    
if __name__ == "__main__":
    main()