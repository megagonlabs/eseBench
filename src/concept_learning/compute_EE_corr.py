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


from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, get_masked_contexts
from utils import LMProbe, LMProbe_GPT2, LMProbe_Joint

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to clusters')
    parser.add_argument('-ng', '--max_allowed_ngrams', type=int, default=3,
                        help='Max length of ngrams (tokenized)')
    parser.add_argument('-ct', '--max_contexts', type=int, default=50,
                        help='Max number of contexts to use')
    parser.add_argument('-top_k', '--top_k', type=int, default=100,
                        help='Top-k to extract for each instance')

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


def _entity_expansion_corr_instance(entity,
                                    contexts,
                                    all_ents_tokenized,
                                    lm_probe,
                                    max_allowed_ngrams,
                                    top_k):
    
    _expand_records = []
    
    entity2probs = defaultdict(list)

    for _context in contexts:
        for n_grams in range(1, max_allowed_ngrams+1):
            _ctxt = _context.replace('[MASK]', '[MASK]' + ' [MASK]' * (n_grams-1))
            _ctxt = '[CLS] ' + _ctxt + ' [SEP]'
            _cands = [e_t for e_t in all_ents_tokenized if len(e_t) == n_grams]
            _cand_scores = lm_probe.score_candidates(_ctxt, _cands)

            for _d in _cand_scores:
                _c = ' '.join(_d['cand']).replace(' ##', '')
                _s = _d['score']
                entity2probs[_c].append(_s)

#     print('entity2probs:', len(entity2probs), len(entity2probs[entity]))
    for _e, _ss in entity2probs.items():
        assert len(_ss) == len(entity2probs[entity]), \
            f'entity: {_e} | {lm_probe.tokenizer.tokenize(_e)}\n\
            len(_ss) = {len(_ss)}\n\
            len(entity2probs["{entity}"]) = {len(entity2probs[entity])}'

    _target_ss = entity2probs[entity]
    _target_ss = _target_ss / np.sum(_target_ss)

#     print(_target_ss.shape, _target_ss)

    mean_l = [(_e, np.mean(_ss)) for _e, _ss in entity2probs.items()]
    mean_l.sort(key=lambda p : p[-1], reverse=True)
    kl_l = [(_e, entropy(_target_ss, _ss)) for _e, _ss in entity2probs.items()]
    kl_l.sort(key=lambda p : p[-1], reverse=False)
    pearson_l = [(_e, pearsonr(_target_ss, _ss)[0]) for _e, _ss in entity2probs.items()]
    pearson_l.sort(key=lambda p : p[-1], reverse=True)

    entity2ranks = defaultdict(list)
    entity2scores = defaultdict(dict)
    for i, (_e, _s) in enumerate(mean_l):
        entity2ranks[_e].append(i)
        entity2scores[_e]["mean"] = _s
    for i, (_e, _s) in enumerate(kl_l):
        entity2ranks[_e].append(i)
        entity2scores[_e]["kl"] = _s
    for i, (_e, _s) in enumerate(pearson_l):
        entity2ranks[_e].append(i)
        entity2scores[_e]["pearson"] = _s
    # To simile top-k set intersection, keep the highest rank of _e among each criteria
    entity_overall_ranks = [(_e, max(_ranks)) for _e, _ranks in entity2ranks.items()]
    entity_overall_ranks.sort(key=lambda p : p[-1])
    entity_overall_ranks_dict = dict(entity_overall_ranks)
    # Now, the top-k is for the final selection, not for each criteria
    sel_entities = [_e for _e, _ in entity_overall_ranks[:top_k]]

    for _e in sel_entities:
#         if (_e in _expand_set) or (_e in seed_instances):
#             continue
#         _expand_set.add(_e)
        _d = dict(entity2scores[_e])
        _d['entity'] = _e
        _d['max_rank'] = entity_overall_ranks_dict[_e]
        _expand_records.append(_d)

    return _expand_records


def entity_expansion_corr(seed_concepts_path,
                          corpus_path,
                          embed_num_path, 
                          max_allowed_ngrams,
                          max_contexts,
                          top_k,
                          dest,
                          **kwargs):
    
    entities, all_contexts = get_masked_contexts(corpus_path, embed_num_path)    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    lm_probe = LMProbe()
    all_ents_tokenized = [tuple(lm_probe.tokenizer.tokenize(e)) for e in entities]
    all_ents_tokenized = list(set(all_ents_tokenized))
    
    
#     if contexts is None:
#         try:
#             contexts = dedup_context[entity]
#         except KeyError:
#             print(f'"{entity}" not an extracted entity!')
#             return None

    _out_records = []

    for i, (a_concept, u_concept, gnrl, seed_instances) in tqdm(seed_concepts_df.iterrows(), total=seed_concepts_df.shape[0]):
        _expand_set = set()
        _expand_records = []
        
        for _inst in seed_instances:
            print(f'{a_concept} :: {_inst}')
            try:
                contexts = all_contexts[_inst]
            except KeyError:
                print(f'"{_inst}" not an extracted entity!')
                continue
            if len(contexts) < 2:
                print(f'"{_inst}" only have {len(contexts)} context')
                continue

            _entity_pieces = lm_probe.tokenizer.tokenize(_inst)
            if len(_entity_pieces) > max_allowed_ngrams:
                print(f'{_entity_pieces} too many word pieces (max {max_allowed_ngrams})')
                continue

        
            _inst_expand_records = _entity_expansion_corr_instance(entity=_inst,
                                                                   contexts=contexts[:max_contexts],
                                                                   all_ents_tokenized=all_ents_tokenized,
                                                                   lm_probe=lm_probe,
                                                                   max_allowed_ngrams=max_allowed_ngrams,
                                                                   top_k=top_k)
            for _d in _inst_expand_records:
                _e = _d['entity']
                if _e in _expand_set or _e in seed_instances:
                    continue
                _expand_set.add(_e)
                _expand_records.append(_d)
            
            # Sort among each concept, although the max_rank are under different seed instances 
            _expand_records.sort(key=lambda d : d['max_rank'])
        
        for _d in _expand_records:
            _out_d = dict()
            _out_d['concept'] = a_concept
            _out_d['neighbor'] = _d['entity']
            _out_d['mean'] = _d['mean']
            _out_d['kl'] = _d['kl']
            _out_d['pearson'] = _d['pearson']
            _out_d['max_rank'] = _d['max_rank']
            _out_records.append(_out_d)
        

    if dest is not None:
        pd.DataFrame(_out_records).to_csv(dest, index=False)
    return _out_records
    
    
def main():
    args = parse_arguments()

    args.seed_concepts_path = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    args.corpus_path = os.path.join(args.dataset_path, 'sentences_with_company.json')
    args.embed_num_path = os.path.join(args.dataset_path, 'BERTembednum+seeds.txt')
    
    entity_expansion_corr(**vars(args))


if __name__ == "__main__":
    main()