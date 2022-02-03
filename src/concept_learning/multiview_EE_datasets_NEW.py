from tqdm.notebook import tqdm
import argparse
import re
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, entropy, gmean
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
from collections import defaultdict, Counter
import time
import importlib
import pytorch_lightning as pl

import logging
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
import os
import sys
import math
from annoy import AnnoyIndex
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
from glob import glob

import spacy
from spacy.matcher import Matcher
from spacy.lang.en import English

from compute_concept_clusters import knn
from compute_keyphrase_embeddings import ensure_tensor_on_device, mean_pooling

from compute_multi_view_embeddings import get_lm_probe_concept_embeddings

from lm_probes import LMProbe, LMProbe_GPT2, LMProbe_Joint, LMProbe_PMI, LMProbe_PMI_greedy
from utils import load_seed_aligned_concepts, load_seed_aligned_relations, load_benchmark
from utils import load_embeddings, load_embeddings_dict, load_EE_labels
from utils import load_EE_labels
from utils import get_masked_contexts, bert_untokenize
from utils import learn_patterns

from roberta_ses.interface import Roberta_SES_Entailment


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--dataset_path', type=str, 
#                         required=True, help='Dataset path with intermediate output')
#     parser.add_argument('-o', '--test_output', type=str, 
#                         required=True, help='Test output path')
#     parser.add_argument('-ep', '--epochs', type=int, 
#                         required=True, help='number of epochs')
#     args = parser.parse_args()
#     return args


class EE_Dataset(Dataset):
    '''
    Base class, merging original [Wiki/Indeed]_EE_Dataset[_2]
    '''

    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
                 lm_ent_using_hearst=True,
                 labels_in_datafile=True,  # True -> old Wiki*, False -> old Indeed* 
                 emb_ent_dict=None,
                 emb_cc_dict=None,
                 lm_ent_dict=None,
                 lm_cc_dict=None,
                 embeddings_dim=768):
        self.ds_path = ds_path
        self.emb_ent_path = emb_ent_path
        self.emb_cc_path = emb_cc_path
        self.lm_ent_path = lm_ent_path
        self.lm_cc_path = lm_cc_path
        self.lm_ent_using_hearst = lm_ent_using_hearst
        self.labels_in_datafile = labels_in_datafile
        self.embeddings_dim = embeddings_dim
        
        # Hard-code for now
        if not lm_ent_using_hearst:
            raise NotImplementedError()
        
        # Load embedding dictionaries (if not given in args)
        self.emb_ent_dict = emb_ent_dict or load_embeddings_dict(emb_ent_path, embeddings_dim)
        self.emb_cc_dict = emb_cc_dict or load_embeddings_dict(emb_cc_path, embeddings_dim)
        self.lm_cc_dict = lm_cc_dict or load_embeddings_dict(lm_cc_path, embeddings_dim)
        if lm_ent_dict is not None:
            self.lm_ent_dict = lm_ent_dict
        elif self.lm_ent_using_hearst:
            lm_ent_df = pd.read_csv(lm_ent_path)
            self.lm_ent_dict = dict()
            for d in lm_ent_df.to_dict('record'):
                k = (d['concept'], d['neighbor'])
                v = [float(v) for v in d['embedding'].split(' ')]
                self.lm_ent_dict[k] = v
        else:
            self.lm_ent_dict = load_embeddings_dict(lm_ent_path, embeddings_dim)
        
        
        # Load samples from datafile
        self._load_sample_records()

    
    def _load_sample_records(self):
        # Load samples from datafile (ds_path)
        if self.labels_in_datafile:
            self.sample_records = pd.read_csv(self.ds_path).to_dict('record')
        else:
            gold_df = pd.read_csv(ds_path)
            _valid_ents = set(self.lm_ent_dict.keys()) & set(self.emb_ent_dict.keys())

            self.sample_records = []
            for d in gold_df.to_dict('record'):
                if d['neighbor'] not in _valid_ents:
                    continue
                self.sample_records.append({
                    'concept': d['concept'],
                    'neighbor': d['neighbor'],
                    'label': 1
                })
            for cc in self.lm_cc_dict.keys():
                _gold_ents = set(gold_df[gold_df['concept'] == cc]['neighbor'])
                for e in _valid_ents:
                    if e in _gold_ents:
                        continue
                    self.sample_records.append({
                        'concept': cc,
                        'neighbor': e,
                        'label': 0
                    })
            self.sample_records.sort(key=lambda d: (d['concept'], -d['label']))    
    
    def __getitem__(self, idx):
        d = self.sample_records[idx]
        _e, _cc, _lbl = d['neighbor'], d['concept'], d['label']
        emb_ent = self.emb_ent_dict[_e]
        emb_cc = self.emb_cc_dict[_cc]
        lm_ent = self.lm_ent_dict[(_cc, _e)]
        lm_cc = self.lm_cc_dict[_cc]
        labels = _lbl
        
        return {
            'emb_ent': torch.tensor(emb_ent, dtype=torch.float32),
            'emb_cc': torch.tensor(emb_cc, dtype=torch.float32),
            'lm_ent': torch.tensor(lm_ent, dtype=torch.float32),
            'lm_cc': torch.tensor(lm_cc, dtype=torch.float32),
            'label': torch.tensor(labels, dtype=torch.int),
        }
    
    def __len__(self):
        return len(self.sample_records)


    
class EE_Dataset_pairs(EE_Dataset):
    '''
    Inheriting EE_Dataset;
    Genrating pos and neg sample pairs 
    '''
    
    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
                 lm_ent_using_hearst=True,
                 labels_in_datafile=True,  # True -> old Wiki*, False -> old Indeed* 
                 emb_ent_dict=None,
                 emb_cc_dict=None,
                 lm_ent_dict=None,
                 lm_cc_dict=None,
                 embeddings_dim=768):
        
        super().__init__(ds_path=ds_path,
                         emb_ent_path=emb_ent_path,
                         emb_cc_path=emb_cc_path,
                         lm_ent_path=lm_ent_path,
                         lm_cc_path=lm_cc_path,
                         lm_ent_using_hearst=lm_ent_using_hearst,
                         labels_in_datafile=labels_in_datafile,
                         emb_ent_dict=emb_ent_dict,
                         emb_cc_dict=emb_cc_dict,
                         lm_ent_dict=lm_ent_dict,
                         lm_cc_dict=lm_cc_dict,
                         embeddings_dim=embeddings_dim)
    
        # pairing
        self._load_sample_pair_records()
        
        
    def _load_sample_pair_records(self):
        # Should be usable to reload (re-matching) sample pairs 
        _cc_pos_entities = defaultdict(set)
        _cc_neg_entities = defaultdict(set)
        for d in self.sample_records:
            cc, e, lbl = d['concept'], d['neighbor'], d['label']
            if lbl == 1:
                _cc_pos_entities[cc].add(e)
            else:
                _cc_neg_entities[cc].add(e)
        
        self.sample_pair_records = []
        for cc, ents in _cc_pos_entities.items():
            neg_ents = list(_cc_neg_entities[cc])
            random.shuffle(neg_ents)
            
            for i, e in enumerate(ents):
                _ne = neg_ents[i % len(neg_ents)]
                self.sample_pair_records.append({
                    'pos_concept': cc,
                    'pos_neighbor': e,
                    'neg_concept': cc,
                    'neg_neighbor': _ne
                })
                
                ## ent-based pair, i.e. vs (e, _ncc), if e not in _cc_pos_entities[_ncc]
                neg_ccs = [_ncc for _ncc in _cc_pos_entities.keys() if e not in _cc_pos_entities[_ncc]]
                _ncc = random.choice(neg_ccs)
                self.sample_pair_records.append({
                    'pos_concept': cc,
                    'pos_neighbor': e,
                    'neg_concept': _ncc,
                    'neg_neighbor': e
                })
    
    
    def __getitem__(self, idx):
        d = self.sample_pair_records[idx]
        _pos_e, _pos_cc, _neg_e, _neg_cc = d['pos_neighbor'], d['pos_concept'], d['neg_neighbor'], d['neg_concept']
        
        emb_pos_ent = self.emb_ent_dict[_pos_e]
        emb_pos_cc = self.emb_cc_dict[_pos_cc]
        lm_pos_ent = self.lm_ent_dict[(_pos_cc, _pos_e)]
        lm_pos_cc = self.lm_cc_dict[_pos_cc]
        
        emb_neg_ent = self.emb_ent_dict[_neg_e]
        emb_neg_cc = self.emb_cc_dict[_neg_cc]
        lm_neg_ent = self.lm_ent_dict[(_neg_cc, _neg_e)]
        lm_neg_cc = self.lm_cc_dict[_neg_cc]
        
        return {
            'emb_pos_ent': torch.tensor(emb_pos_ent, dtype=torch.float32),
            'emb_pos_cc': torch.tensor(emb_pos_cc, dtype=torch.float32),
            'lm_pos_ent': torch.tensor(lm_pos_ent, dtype=torch.float32),
            'lm_pos_cc': torch.tensor(lm_pos_cc, dtype=torch.float32),
            'emb_neg_ent': torch.tensor(emb_neg_ent, dtype=torch.float32),
            'emb_neg_cc': torch.tensor(emb_neg_cc, dtype=torch.float32),
            'lm_neg_ent': torch.tensor(lm_neg_ent, dtype=torch.float32),
            'lm_neg_cc': torch.tensor(lm_neg_cc, dtype=torch.float32),
        }
    
    def __len__(self):
        return len(self.sample_pair_records)
    
    