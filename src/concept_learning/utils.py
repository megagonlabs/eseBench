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
import spacy
from spacy.matcher import Matcher
import networkx as nx
import itertools
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


def load_EE_labels(ee_labels_path):
    ee_labels_df = pd.read_csv(ee_labels_path)
    concept_list = list(set(ee_labels_df['concept'].tolist()))

    ee_labels_dict = dict()
    for _cc in concept_list:
        ee_labels_dict[_cc] = ee_labels_df[ee_labels_df['concept'] == _cc]['neighbor'].tolist()

    return ee_labels_dict


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


def learn_patterns(all_mentions, src, tgt, nlp=None):
    if nlp is None:
        nlp = spacy.load('en_core_web_sm')
    
    src_matcher = Matcher(nlp.vocab)
    src_pattern = [{"LOWER": t} for t in src.split(' ')]
    src_matcher.add("src", [src_pattern])

    tgt_matcher = Matcher(nlp.vocab)
    tgt_pattern = [{"LOWER": t} for t in tgt.split(' ')]
    tgt_matcher.add("tgt", [tgt_pattern])
    
    patterns = {}
    for mention in all_mentions:
        doc = nlp(mention)
        src_matches = src_matcher(doc)
        if len(src_matches) == 0:
            # print('src not matched')
            continue
        tgt_matches = tgt_matcher(doc)
        if len(tgt_matches) == 0:
            # print('tgt not matched')
            continue
        src_match = src_matches[0]
        tgt_match = tgt_matches[0]
        src_span = doc[src_match[1]: src_match[2]]
        tgt_span = doc[tgt_match[1]: tgt_match[2]]
        
        if len(spacy.util.filter_spans([src_span, tgt_span])) != 2: # distinct_spans
            print('overlapping spans')
            continue
        
        src_root = src_span.root
        tgt_root = tgt_span.root
        
        #  print(mention)
        edges = []
        for token in doc:
            for child in token.children:
                edges.append(('{}-{}'.format(token.lower_,token.i), '{}-{}'.format(child.lower_,child.i))) 
        
        graph = nx.Graph(edges) 
        path = None
        source = '{}-{}'.format(src_root.lower_, src_root.i)
        target = '{}-{}'.format(tgt_root.lower_, tgt_root.i)
        
        try:
            assert nx.has_path(graph, source=source, target=target)
        except:
            continue
            
        path = nx.shortest_path(graph, source=source, target=target)
            
        #  print(path)
        if path is not None:
            for t in src_span:
                n = '{}-{}'.format(t.lower_, t.i)  
                if n not in path:
                    path.append(n)
            for t in tgt_span:
                n = '{}-{}'.format(t.lower_, t.i)
                if n not in path:
                    path.append(n)
            path_nodes = {}
            for p in path:
                t, i = p.rsplit('-', 1)
                i = int(i)
                if i in range(src_match[1], src_match[2]):
                    t = '<src>'
                elif i in range(tgt_match[1], tgt_match[2]):
                    t = '<tgt>'
                path_nodes[i] = t
            path_nodes = sorted(path_nodes.items(), key=lambda x: x[0])
            pattern = ' '.join([p[1] for p in path_nodes])
#             patterns[pattern] = patterns.get(pattern, 0) + 1
            patterns[pattern] = patterns.get(pattern, []) + [mention]
    patterns = {k:v for k,v in patterns.items() if len(v) > 1}
    patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
    return patterns



def main():
    pass
    
    
if __name__ == "__main__":
    main()