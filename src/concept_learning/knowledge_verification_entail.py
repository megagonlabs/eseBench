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

from compute_concept_clusters import load_embeddings

from roberta_ses.interface import Roberta_SES_Entailment

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-r', '--roberta_dir', type=str,
                        required=True, help='Roberta directory path')
    parser.add_argument('-rs', '--roberta_ses_path', type=str,
                        required=True, help='Path to Roberta-SES checkpoint')
    parser.add_argument('-o', '--dest', type=str, required=True,
                        help='Path to output file')
#     parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
#                         help='embedding_dim')
    parser.add_argument('-p', '--p_thres', type=float, default=0.7,
                        help='Threshold to keep the evidence')

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

def find_evidences_RE(relations_path,
                      corpus_path,
                      roberta_dir,
                      roberta_ses_path,
                      p_thres,
                      dest,
                      **kwargs):
    
    _pos_templates = [
        '{0} allows {1}',
        '{0} requires {1}',
    ]

    _neg_templates = [
        '{0} doesn\'t allow {1}',
        '{0} doesn\'t require {1}',
    ]
    
    print('Loading files...')
    
    df_relations = pd.read_csv(relations_path)
    sent_dicts = []
    with open(corpus_path, 'r') as f:
        sent_dicts = [json.loads(l) for l in f.readlines()]
    
    entailment_model = Roberta_SES_Entailment(roberta_path=roberta_dir,
        ckpt_path=os.path.join(roberta_ses_path),
        max_length=512,
        device_name='cpu')
    
    # Dict[Tuple(head, rel, tail): List[Tuple(s1(evid), s2(tmpl), score)]]
    pos_evidences = defaultdict(list)
    neg_evidences = defaultdict(list)
    
    # collect all relations 
    rels = []
    head2rels = defaultdict(list)
    tail2rels = defaultdict(list)
    for i, row in df_relations.iterrows():
        _h = row['head']
        _t = row['tail']
        _r = 'has_dress_code'
        rels.append((_h, _r, _t))
        if row['base'] == 'HEAD':
            head2rels[_h].append((_h, _r, _t))
        else:
            tail2rels[_t].append((_h, _r, _t))

    # collect sents for each entity 
    entity2sents = defaultdict(set)
    for i, d in enumerate(sent_dicts):
        _s = f"{d['company']} : {' '.join(d['tokens'])}".lower()
        for _e in d['entities']:
            entity2sents[_e].add(_s)
    
    for _h, _r, _t in tqdm(rels, desc='Finding evidence for rels'):
#         # assure key existence
#         _ = pos_evidences[(_h, _r, _t)]
#         _ = neg_evidences[(_h, _r, _t)]
        
        h_sents = entity2sents[_h]
        t_sents = entity2sents[_t]
        intersect_sents = h_sents & t_sents
        
        for _s in intersect_sents:
            _ss = _s.strip()

            # Try all pos/neg relation templates, save the best template  
            _max_pos_ev = (None, None, 0)
            for _tmpl in _pos_templates:
                _tmpl_filled = _tmpl.format(_h, _t)
                _entail_pred, _entail_probs = entailment_model.predict(_ss, _tmpl_filled)
                _entail_score = _entail_probs[2].item()
                if _entail_score > _max_pos_ev[-1]:
                    _max_pos_ev = (_ss, _tmpl_filled, _entail_score)

            _max_neg_ev = (None, None, 0)
            for _tmpl in _neg_templates:
                _tmpl_filled = _tmpl.format(_h, _t)
                _entail_pred, _entail_probs = entailment_model.predict(_ss, _tmpl_filled)
                _entail_score = _entail_probs[2].item()
                if _entail_score > _max_neg_ev[-1]:
                    _max_neg_ev = (_ss, _tmpl_filled, _entail_score)

            if _max_pos_ev[-1] > p_thres:
                pos_evidences[(_h, _r, _t)].append(_max_pos_ev)
            if _max_neg_ev[-1] > p_thres:
                neg_evidences[(_h, _r, _t)].append(_max_neg_ev)
    
    out_list = []
    for _rel in rels:
        _pos_evs = pos_evidences[_rel]
        _neg_evs = neg_evidences[_rel]
        if len(_pos_evs) == len(_neg_evs) == 0:
            continue
        _pos_evs.sort(key=lambda p : p[-1], reverse=True)
        _neg_evs.sort(key=lambda p : p[-1], reverse=True)
        out_list.append({
            'relation': _rel,
            'pos_evidences': _pos_evs,
            'neg_evidences': _neg_evs,
        })
    
    with open(dest, 'w') as f:
        for d in out_list:
            f.write(json.dumps(d) + '\n')
    
#     for _rel, _evidences in pos_evidences.items():
#         _evidences.sort(key=lambda p : p[-1], reverse=True)
#     for _rel, _evidences in neg_evidences.items():
#         _evidences.sort(key=lambda p : p[-1], reverse=True)
    
    return out_list
    
    
def main():
    args = parse_arguments()
    args.relations_path = os.path.join(args.dataset_path, 'rel_extraction.csv')
    args.corpus_path = os.path.join(args.dataset_path, 'sentences_with_company.json')

    find_evidences_RE(**vars(args))
    
    
if __name__ == "__main__":
    main()