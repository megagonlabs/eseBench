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
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from roberta_ses.interface import Roberta_SES_Entailment


def load_embeddings(embed_src, embedding_dim):
    with open(embed_src, 'r') as fin:
        lines = fin.readlines()
        lines = [l.strip() for l in lines]

    embeddings = {}
    for line in lines:
        tmp = line.split(' ')
        if len(tmp) < embedding_dim + 1:
            continue
        vec = tmp[-embedding_dim:]
        vec = [float(v) for v in vec]
        entity = ' '.join(tmp[:(len(tmp) - embedding_dim)])
        embeddings[entity] = vec
    df = pd.DataFrame(embeddings.items(), columns=['entity', 'embedding'])
    return df

def load_seed_aligned_concepts(path):
    df = pd.read_csv(path)
    df = df[df["generalizations"] != "x"]
    df["seedInstances"] = df["seedInstances"].map(lambda s : eval(str(s)))
    return df

def load_seed_aligned_relations(path):
    df = pd.read_csv(path)
    df = df[df["range"] != "x"]
    return df


def load_benchmark(benchmark_full_path,
                   seed_concepts_path,
                   seed_relations_path):
    benchmark = pd.read_csv(benchmark_full_path)
    concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    relations_df = load_seed_aligned_relations(seed_relations_path)
    
    concepts_dict = dict(zip(concepts_df['alignedCategoryName'].tolist(), concepts_df.to_dict('records')))
    relations_dict = dict(zip(relations_df['alignedRelationName'].tolist(), relations_df.to_dict('records')))
    
    # Dict[str(_type), Set[str(_e)]]
    all_concepts = defaultdict(set)
    # Dict[str(_r), Set[Tuple(_h, _r, _t)]]
    all_rel_tuples = defaultdict(set)
    
    for i, row in benchmark.iterrows():
        if row['type'] != 'fact':
            continue
        
        _r = row['relation_name']
        _h_type = row['n_head_category']
        _t_type = row['n_tail_category']
        
        if _r not in relations_dict:
            continue
        _relation_row = relations_dict[_r]
        if _relation_row['domain'] != _h_type or _relation_row['range'] != _t_type:
            continue
        
        row_n_head = str(row['n_head']).lower()
        row_n_tail = str(row['n_tail']).lower()
        
        if row_n_head == 'company':
            evidence_sents = eval(str(row['sentences']))
            head_instances = eval(str(row['Evidence']))
            assert len(evidence_sents) == len(head_instances), f'Line {i} length mismatch'

            for inst in head_instances:
                if len(inst) > 0:
                    all_concepts[_h_type].add(inst.lower())
                    all_concepts[_t_type].add(row_n_tail)
                    all_rel_tuples[_r].add(
                        (inst.lower(), _r, row_n_tail)
                    )
        else:
            # treat n_head directly as instance 
            all_concepts[_h_type].add(row_n_head)
            all_concepts[_t_type].add(row_n_tail)
            all_rel_tuples[_r].add(
                (row_n_head, _r, row_n_tail)
            )
        
    return all_concepts, all_rel_tuples


def get_masked_contexts(corpus_path, embed_num_path):
    """Return a (list of) sentence(s) with entity replaced with MASK."""
    
    with open(corpus_path, 'r') as f:
        sent_dicts = [json.loads(l) for l in f.readlines()]
    with open(embed_num_path, 'r') as f:
        entities = [l.rsplit(' ', 1)[0] for l in f.readlines()]
    
    entity2sents = defaultdict(set)
    for i, d in enumerate(sent_dicts):
        _s = f" {' '.join(d['tokens'])} ".lower()
        for _e in d['entities']:
            _e_pat = f" {_e} "
            if _s.count(_e_pat) != 1:
                # 0 = implicit company name; 2+ = multiple mentions 
                continue
            _s_masked = _s.replace(_e_pat, " [MASK] ")
            _s_masked = _s_masked.strip()
            entity2sents[_e].add(_s_masked)
    
    dedup_context = dict()
    for _e, _v in entity2sents.items():
        dedup_context[_e] = list(_v)

    return entities, dedup_context


def bert_untokenize(pieces):
    return ' '.join(pieces).replace(' ##', '')


def main():
    pass
    
    
if __name__ == "__main__":
    main()