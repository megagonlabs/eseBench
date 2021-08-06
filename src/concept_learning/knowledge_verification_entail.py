from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd
import os
import torch
import math
import json
from collections import defaultdict
from annoy import AnnoyIndex

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations

from roberta_ses.interface import Roberta_SES_Entailment

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
#     parser.add_argument('-b', '--benchmark_path', type=str,
#                         required=True, help='Benchmark directory path')
    parser.add_argument('-in', '--input_re_path', type=str,
                        required=True, help='RE input file path')
    parser.add_argument('-r', '--roberta_dir', type=str,
                        required=True, help='Roberta directory path')
    parser.add_argument('-rs', '--roberta_ses_path', type=str,
                        required=True, help='Path to Roberta-SES checkpoint')
    parser.add_argument('-o_kv', '--dest_kv', type=str, required=True,
                        help='Path to KV output file')
    parser.add_argument('-o_re', '--dest_re', type=str, required=True,
                        help='Path to RE output file, filtered with this KV process')
#     parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
#                         help='embedding_dim')
    parser.add_argument('-p_kv', '--kv_p_thres', type=float, default=0.7,
                        help='Threshold to keep the evidence')
    parser.add_argument('-p_re', '--re_p_thres', type=float, default=0.9,
                        help='Threshold to keep the relation in prediction')
    parser.add_argument('--fast_skip', type=int, default=None,
                        help='If set, only aim to find this many evidences > p_re')

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

def find_evidences_RE(input_re_path,
                      corpus_path,
                      roberta_dir,
                      roberta_ses_path,
                      templates_path,
                      kv_p_thres,
                      re_p_thres,
                      dest_kv,
                      dest_re,
                      kv_collection_path,
                      fast_skip,
                      **kwargs):

    print('Loading files...')
    
    with open(templates_path, 'r') as f:
        all_templates = json.load(f)
    
    df_relations = pd.read_csv(input_re_path)
    sent_dicts = []
    with open(corpus_path, 'r') as f:
        sent_dicts = [json.loads(l) for l in f.readlines()]
    
    with open(kv_collection_path, 'r') as f:
        kv_collections = [json.loads(l) for l in f.readlines()]
    kv_collections_dict = dict([(tuple(d['relation']), d) for d in kv_collections])
#     new_relations = []  # relations not in kv_collections, either verified or not 
    
    entailment_model = Roberta_SES_Entailment(roberta_path=roberta_dir,
        ckpt_path=os.path.join(roberta_ses_path),
        max_length=512,
        device_name='cpu')
    
    # Dict[Tuple(head, rel, tail): List[Tuple(s1(evid), s2(tmpl), score)]]
    pos_evidences = defaultdict(list)
    neg_evidences = defaultdict(list)
    
    # collect all relations 
    rels = []
#     head2rels = defaultdict(list)
#     tail2rels = defaultdict(list)
    for i, row in df_relations.iterrows():
        _h = row['head']
        _t = row['tail']
        _r = row['relation']
        rels.append((_h, _r, _t))

    # collect sents for each entity 
    entity2sents = defaultdict(set)
    for i, d in enumerate(sent_dicts):
        _s = f"{d['company']} : {' '.join(d['tokens'])}".lower()
        for _e in d['entities']:
            entity2sents[_e].add(_s)
    
    for _h, _r, _t in tqdm(rels, desc='Finding evidence for rels'):
        if _h == _t:  # such cases (_h == _t) are unlikely right and take a lot of time
            continue
        
        if (_h, _r, _t) in kv_collections_dict:  # Already in kv_collections 
            # Take from kv_collections 
            _rel = (_h, _r, _t)
            _d = kv_collections_dict[_rel]
            pos_evidences[_rel] = _d['pos_evidences']
            neg_evidences[_rel] = _d['neg_evidences']
            continue

        _pos_templates = all_templates[_r]['positive']
        _neg_templates = all_templates[_r]['negative']
        
        h_sents = entity2sents[_h]
        t_sents = entity2sents[_t]
        intersect_sents = h_sents & t_sents
        
        good_ev_cnt = 0
        
        for _s in intersect_sents:
            if (fast_skip is not None) and (good_ev_cnt >= fast_skip):
                break
                
            _ss = _s.strip()

            # Try all pos/neg relation templates, save the best template  
            _max_pos_ev = (None, None, 0)
            for _tmpl in _pos_templates:
                _tmpl_filled = _tmpl.format(_h, _t)
                _entail_pred, _entail_probs = entailment_model.predict(_ss, _tmpl_filled)
                _entail_score = _entail_probs[2].item()
                if _entail_score > _max_pos_ev[-1]:
                    _max_pos_ev = (_ss, _tmpl_filled, _entail_score)
#                     max_score = max(max_score, _entail_score)

            _max_neg_ev = (None, None, 0)
            for _tmpl in _neg_templates:
                _tmpl_filled = _tmpl.format(_h, _t)
                _entail_pred, _entail_probs = entailment_model.predict(_ss, _tmpl_filled)
                _entail_score = _entail_probs[2].item()
                if _entail_score > _max_neg_ev[-1]:
                    _max_neg_ev = (_ss, _tmpl_filled, _entail_score)
#                     max_score = max(max_score, _entail_score)

            if _max_pos_ev[-1] > kv_p_thres:
                pos_evidences[(_h, _r, _t)].append(_max_pos_ev)
            if _max_neg_ev[-1] > kv_p_thres:
                neg_evidences[(_h, _r, _t)].append(_max_neg_ev)
            if max(_max_pos_ev[-1], _max_neg_ev[-1]) > re_p_thres:
                good_ev_cnt += 1
    
    out_kv_list = []
    out_collection_list = []
    out_rels_list = []
    for _rel in rels:
#         if _rel not in kv_collections_dict:
#             new_relations.append(_rel)
        
        _pos_evs = pos_evidences[_rel]
        _neg_evs = neg_evidences[_rel]
        
        _pos_evs.sort(key=lambda p : p[-1], reverse=True)
        _neg_evs.sort(key=lambda p : p[-1], reverse=True)
        
        _kv_d = {
            'relation': _rel,
            'pos_evidences': _pos_evs,
            'neg_evidences': _neg_evs,
        }
        
        if _rel not in kv_collections_dict:
            # not in collection, add to there (even if empty) 
            out_collection_list.append(_kv_d)
        
        if len(_pos_evs) == len(_neg_evs) == 0:
            # empty, don't add to current output
            continue
        
        # non-empty, add to current output
        out_kv_list.append(_kv_d)
        
        if (len(_pos_evs) > 0 and _pos_evs[0][-1] > re_p_thres) or (len(_neg_evs) > 0 and _neg_evs[0][-1] > re_p_thres):
            # have good evidence, verified, add to relations output 
            out_rels_list.append({
                'head': _rel[0],
                'relation': _rel[1],
                'tail': _rel[2]
            })
    
    with open(kv_collection_path, 'a') as f:
        for d in out_collection_list:
            f.write(json.dumps(d) + '\n')
    
    with open(dest_kv, 'w') as f:
        for d in out_kv_list:
            f.write(json.dumps(d) + '\n')
    
    out_rels_df = pd.DataFrame(out_rels_list)
    out_rels_df.to_csv(dest_re, index=None)
    
#     for _rel, _evidences in pos_evidences.items():
#         _evidences.sort(key=lambda p : p[-1], reverse=True)
#     for _rel, _evidences in neg_evidences.items():
#         _evidences.sort(key=lambda p : p[-1], reverse=True)
    
    return out_kv_list
    
    
def main():
    args = parse_arguments()
#     args.input_re_path = os.path.join(args.dataset_path, 'rel_extraction_RE.csv')
    args.corpus_path = os.path.join(args.dataset_path, 'sentences_with_company.json')
    args.kv_collection_path = os.path.join(args.dataset_path, 'kv_evidences_collection.json')
    args.templates_path = 'templates_manual.json'

    find_evidences_RE(**vars(args))
    
    
if __name__ == "__main__":
    main()