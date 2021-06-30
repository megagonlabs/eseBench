from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cosine
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

from compute_concept_clusters import load_embeddings

from roberta_ses.interface import Roberta_SES_Entailment

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-cknn', '--concept_knn_path', type=str,
                        required=True, help='Path to concept knn file')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to output file')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')
    parser.add_argument('-topk', '--topk', type=int, default=300,
                        help='The number of entities to keep for each base head/tail')

    args = parser.parse_args()
    return args

def load_seed_aligned_concepts(path):
    df = pd.read_csv(path)
    df = df[df["generalizations"] != "x"]
    df["seedInstances"] = df["seedInstances"].map(lambda s : eval(str(s)))
    return df

def load_seed_aligned_relations(path):
    df = pd.read_csv(path)
    df = df[df["range"] != "x"]
    return df



class LMProbe(object):
    def __init__(self, model_name='bert-base-uncased', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

    def fill_multi_mask(self, input_txt, topk=3):
        if not (input_txt.startswith('[CLS]') and input_txt.endswith('[SEP]')):
            raise Exception('Input string must start with [CLS] and end with [SEP]')
        if not '[MASK]' in input_txt:
            raise Exception('Input string must have at least one mask token')
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        sorted_probs = sorted_probs.detach().cpu().numpy()
        sorted_idx = sorted_idx.detach().cpu().numpy()

        masked_cands = []
        for k in range(topk):
            predicted_indices = [sorted_idx[i, k].item() for i in mask_indices]
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices)
            predicted_probs = [sorted_probs[i, k].item() for i in mask_indices]
            seq = []
            for token_id, token, prob, masked_index in zip(predicted_indices, predicted_tokens, predicted_probs,
                                                           mask_indices):
                seq.append({"token": token_id, "token_str": token, "prob": prob, "masked_pos": masked_index})
            masked_cands.append(seq)

        return masked_cands
    
    def score_candidates(self, input_txt, cands):
        # cands: List[List[str]], list of tokenized candidates 
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        
        if tokenized_txt[0] != "[CLS]" or tokenized_txt[-1] != "[SEP]":
            raise Exception(f'Input string must start with [CLS] and end with [SEP], got {input_txt}')
        if "[MASK]" not in tokenized_txt:
            raise Exception(f'Input string must have at least one mask token, got {input_txt}')
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        probs = probs.detach().cpu().numpy()

        cand_scores = []
        for c in cands:
            assert len(c) == len(mask_indices), f'cand {c}; len(mask_indices) = {len(mask_indices)}'

            _scores = []
            c_token_ids = self.tokenizer.convert_tokens_to_ids(c)
            for i, token_id in zip(mask_indices, c_token_ids):
                _scores.append(probs[i, token_id].item())
            score = np.prod(_scores)
            cand_scores.append({"cand": c, "score": score})

        cand_scores.sort(key=lambda d : d["score"], reverse=True)
        return cand_scores
    


def direct_probing_RE_v3(seed_aligned_concepts_path,
                         seed_aligned_relations_path,
                         emb_path,
                         concept_knn_path,
                         templates_path,
                         lm_probe=None,
                         embedding_dim=768,
                         scores_agg_func=None,
                         topk=10,
                         dest=None,
                         **kwargs):
    '''
    For each head / tail, rank candidate tails / heads by overall scores. 
    Current (default) overall score: 0.1 * ht_sim + 10 * concept_sim + 0.1 * log(lm_prob)
    '''
    
    seed_concepts_df = load_seed_aligned_concepts(seed_aligned_concepts_path)
#     seed_relations_df = pd.read_csv(seed_relations_path)
#     seed_relations_df = seed_relations_df.iloc[1]
    entity_embeddings = load_embeddings(emb_path, embedding_dim)
    entity_emb_dict = dict(zip(entity_embeddings['entity'].tolist(),
                               entity_embeddings['embedding'].tolist()))
    concept_knn_results = pd.read_csv(concept_knn_path)
    
    with open(templates_path, 'r') as f:
        all_templates = json.load(f)
    templates = all_templates['has_dress_code']
    templates = templates['positive'] + templates['negative']

    if lm_probe is None:
        lm_probe = LMProbe()
    if scores_agg_func is None:
        scores_agg_func = lambda ht_sim, concept_sim, lm_prob : 0.1 * ht_sim + 10 * concept_sim + 0.1 * np.log10(lm_prob)
    
#     head_type = seed_relations_df['domain']
#     tail_type = seed_relations_df['range']
    ## TODO: expand to all relations 
    head_type = "company"
    tail_type = "dress_code"
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

            extraction_results.append({'head': seed_head, 'tail': e_tail, 'base': 'HEAD',
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
        
            extraction_results.append({'head': e_head, 'tail': seed_tail, 'base': 'TAIL',
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


    
def main():
    args = parse_arguments()
    args.seed_aligned_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.seed_aligned_relations_path = os.path.join(args.benchmark_path, 'seed_aligned_relations.csv')
    args.emb_path = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
    args.templates_path = 'templates_manual.json'
    
#     args.dest = os.path.join(args.dataset_path, 'rel_extraction.csv')

    direct_probing_RE_v3(**vars(args))
    
    
    '''
    seed_concepts_path = os.path.join(base_dir, f'data/indeed-benchmark/seed_concepts.csv')
    seed_relations_path = os.path.join(base_dir, f'data/indeed-benchmark/seed_relations.csv')
    seed_aligned_concepts_path = os.path.join(base_dir, f'data/indeed-benchmark/seed_aligned_concepts.csv')
    seed_aligned_relations_path = os.path.join(base_dir, f'data/indeed-benchmark/seed_aligned_relations.csv')
    # knn_path = os.path.join(base_dir, f'data/{data_ac}/intermediate/knn_{cluster_size}.csv')
    concept_knn_path = os.path.join(base_dir, f'data/{data_ac}/intermediate/concept_knn_1000.csv')
    bert_emb_path = os.path.join(base_dir, f'data/{data_ac}/intermediate/BERTembed+seeds.txt')

    extraction_save_path = os.path.join(base_dir, f'data/{data_ac}/intermediate/rel_extraction.csv')
    # extraction_save_path = None

    extraction_results = direct_probing_RE_v3(seed_aligned_concepts_path=seed_aligned_concepts_path,
                                              seed_aligned_relations_path=seed_aligned_relations_path,
                                              emb_path=bert_emb_path,
                                              concept_knn_path=concept_knn_path,
                                              templates=has_dress_code_templates,
                                              lm_probe=lm_probe,
                                              topk=300,
                                              save_path=extraction_save_path)
    '''
    
    
if __name__ == "__main__":
    main()