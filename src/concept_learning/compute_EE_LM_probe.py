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
    parser.add_argument('-d', '--dataset_dir', type=str, default=None,
                        help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_dir', type=str, default=None,
                        help='Benchmark directory path')
    parser.add_argument('-e', '--emb_path', type=str, default=None,
                        required=False, help='Dataset path with pre-computed embeddings')
    parser.add_argument('-s', '--seed_concepts_path', type=str, default=None,
                        help='including auxiliary concepts')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to clusters')
    parser.add_argument('-lm', '--lm_probe_type', type=str,
                        required=True, help='The type of lm_probe to use')
    parser.add_argument('-lm_model', '--lm_probe_model', type=str,
                        default=None, help='The model (e.g. "bert-base-uncased") for lm_probe')
    parser.add_argument('-ng', '--max_n_grams', type=int, default=5,
                        help='Max length of ngrams (tokenized)')
    parser.add_argument('-s_size', '--seed_sample_size', type=int, default=None,
                        help='Size of seed subset used in prediction. None (default) means no subset sampling')
    parser.add_argument('-s_times', '--seed_sample_times', type=int, default=1,
                        help='How many subsets to use (sample) in prediction.')
    parser.add_argument('-agg', '--template_agg', default='max', choices=['max', 'avg'],
                        help='How to aggregate scores from each template')
    parser.add_argument('-topk', '--topk', type=int, default=None,
                        help='Top-k to extract for each concept')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')

    args = parser.parse_args()
    return args


def EE_LMProbe(seed_concepts_path,
               emb_path,
               lm_probe_type,
               lm_probe_model,
               tmpl_agg_func,
               concepts=None,
               embedding_dim=768,
               max_n_grams=5,
               seed_sample_size=None,
               seed_sample_times=1,
               topk=None,
               dest=None,
               **kwargs):
    '''
    EE using LM probing with Hearst prompts like:
    "dress code, such as jeans, [MASK] and shirts." or 
    "dress code, including jeans, [MASK] and shirts." or
    "jeans, [MASK], shirts and other dress code" or 
    
    seed_sample_size, seed_sample_times: for seed self-sampling (each time random sample x seeds, do y times, average)
    '''
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    entity_embeddings = load_embeddings(emb_path, embedding_dim)
    all_entities = entity_embeddings['entity'].tolist()
    all_embeddings = entity_embeddings['embedding'].tolist()
    entity_emb_dict = dict(zip(all_entities, all_embeddings))
    
#     with open(templates_path, 'r') as f:
#         all_templates = json.load(f)
#     templates = all_templates[relation]
#     templates = templates['positive'] + templates['negative']
    
    probe_prompts = [
        "{0}, such as {1}, {2} and {3}.",
        "{0}, including {1}, {2} and {3}.",
        "{1}, {2}, {3} and other {0}.",
    ]

    if lm_probe_type == 'bert':
        print("lm_probe_type='bert' is deprecated, use 'mlm'")
        if lm_probe_model is None:
            lm_probe = LMProbe(max_n_grams=max_n_grams)
        else:
            lm_probe = LMProbe(max_n_grams=max_n_grams, model_name=lm_probe_model)
    elif lm_probe_type == 'mlm':
        if lm_probe_model is None:
            lm_probe = LMProbe(max_n_grams=max_n_grams)
        else:
            lm_probe = LMProbe(max_n_grams=max_n_grams, model_name=lm_probe_model)
    elif lm_probe_type == 'gpt2':
        if lm_probe_model is None:
            lm_probe = LMProbe_GPT2()
        else:
            lm_probe = LMProbe_GPT2(model_name=lm_probe_model)
    elif lm_probe_type == 'joint':
        if lm_probe_model is None:
            lm_probe = LMProbe_Joint(max_n_grams=max_n_grams)
        else:
            ## TODO: bert_model_name and gpt2_model_name
            lm_probe = LMProbe_Joint(max_n_grams=max_n_grams, bert_model_name=lm_probe_model)
#     elif lm_probe_type == 'pmi':
#         lm_probe = LMProbe_PMI()
#     elif lm_probe_type == 'pmi_greedy':
#         lm_probe = LMProbe_PMI_greedy()
    else:
        raise NotImplementedError(f"lm_probe_type = {lm_probe_type}")
    
    if lm_probe_type in ['bert', 'mlm', 'joint']:
        all_cand_entities = [e for e in all_entities if len(lm_probe.tokenizer.tokenize(e)) <= max_n_grams]
    else:
        all_cand_entities = all_entities
    all_cand_entities = list(set(all_cand_entities))
    
    if tmpl_agg_func is None:
        tmpl_agg_func = max
    
    seed_instances_dict = dict(zip(
        seed_concepts_df['alignedCategoryName'].tolist(),
        seed_concepts_df['seedInstances'].tolist()
    ))
    
    if concepts is None:
        concepts = seed_concepts_df['alignedCategoryName'].tolist()
    
    all_extraction_results = []
    for cc in tqdm(concepts):
        # c_head_tokenized = lm_probe.tokenizer.tokenize(c_head)
        seeds = seed_instances_dict[cc]
        cc_phrase = ' '.join(cc.split('_'))
        
        if (seed_sample_size is None) or (seed_sample_size >= len(seeds)):
            seed_subsets = [seeds]
        else:
            # subset sampling 
            assert seed_sample_size >= 2, seed_sample_size
            assert seed_sample_times >= 1, seed_sample_times
            seed_subsets = [random.sample(seeds, k=seed_sample_size) for _ in range(seed_sample_times)]

        extraction_results = []
        cand_scores = []  ## List[Dict["cand": str, "score": float]], the avg score of "cand" over templates & subsets 
        
        if lm_probe_type in ['bert', 'mlm', 'gpt2', 'joint']:
            ## Score: prob
            
            ## List[List[Dict["cand": str, "score": float]]], on each template, the avg score of each cand over subsets
            cand_scores_per_template = []  
            
            for template in probe_prompts:
                ## List[List[Dict["cand": str, "score": float]]], on each subset, the score of each cand
                _cand_scores_per_subset = []  
                for _seeds in seed_subsets:
                    ## template: {0} = concept, {1-3} = instances
                    _input_txt = template.format(cc_phrase, ', '.join(_seeds[:-1]), '[MASK]', _seeds[-1])
                    # print(_input_txt)
                    _cand_scores = lm_probe.score_candidates(_input_txt, all_cand_entities)
                    _cand_scores.sort(key=lambda d : d["cand"])
                    _cand_scores_per_subset.append(_cand_scores)
                
                ## List[Dict["cand": str, "score": float]], the score of each cand (on this template)
                _cand_scores = []
                for _cand_score_lst in zip(*_cand_scores_per_subset):
                    ## _cand_score_lst: List[Dict["cand": str, "score": float]], on each subset, the score of "cand"
                    _cand = _cand_score_lst[0]["cand"]
                    assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
                    _score = sum([d["score"] for d in _cand_score_lst]) / len(_cand_score_lst)
                    # _score = np.log(_score)
                    _cand_scores.append({"cand": _cand, "score": _score})
                    
                cand_scores_per_template.append(_cand_scores)

            for _cand_score_lst in zip(*cand_scores_per_template):
                ## _cand_score_lst: List[Dict["cand": str, "score": float]], on each template, the score of "cand" 
                _cand = _cand_score_lst[0]["cand"]
                assert all(d["cand"] == _cand for d in _cand_score_lst), _cand_score_lst
                _score = tmpl_agg_func([d["score"] for d in _cand_score_lst])
                # _score = np.log(_score)
                cand_scores.append({"cand": _cand, "score": _score})
        elif lm_probe_type in ['pmi', 'pmi_greedy']:
            ## Score: PMI 
            raise NotImplementedError
            

        for d in cand_scores:
            e = d["cand"]
            if e in seeds:
                continue

            lm_score = d["score"]
            extraction_results.append({'concept': cc,
                                       'neighbor': e,
                                       'lm_score': lm_score
                                      })
        
        extraction_results.sort(key=lambda d : d['lm_score'], reverse=True)
        all_extraction_results.extend(extraction_results[:topk])

#     all_extraction_results.sort(key=lambda d : d['overall_score'], reverse=True)
#     all_extraction_results = all_extraction_results[:topk]

    results_df = pd.DataFrame(all_extraction_results)
    if dest is not None:
        results_df.to_csv(dest, index=None)
    return results_df

    
    
def main():
    args = parse_arguments()

    if args.seed_concepts_path is None:
        args.seed_concepts_path = os.path.join(args.benchmark_dir, 'seed_aligned_concepts.csv')
#     if args.aux_concepts:
#         args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts_aux.csv')
#     else:
#         args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
        
#     args.corpus_path = os.path.join(args.dataset_path, 'sentences_with_company.json')
    if args.emb_path is None:
        args.emb_path = os.path.join(args.dataset_dir, 'BERTembed+seeds.txt')
    
    if args.template_agg == 'max':
        args.tmpl_agg_func = lambda l : max(l)
    elif args.template_agg == 'avg':
        args.tmpl_agg_func = lambda l : (sum(l) / len(l))
    
    random.seed(123)
    EE_LMProbe(**vars(args))


if __name__ == "__main__":
    main()