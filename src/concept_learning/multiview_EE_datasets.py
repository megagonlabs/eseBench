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
from torch.utils.data import DataLoader, Dataset
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


class Wiki_EE_Dataset(Dataset):
    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
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
        self.embeddings_dim = embeddings_dim
        
        self.sample_records = pd.read_csv(ds_path).to_dict('record')
        
        self.emb_ent_dict = emb_ent_dict or load_embeddings_dict(emb_ent_path, embeddings_dim)
        self.emb_cc_dict = emb_cc_dict or load_embeddings_dict(emb_cc_path, embeddings_dim)
        self.lm_ent_dict = lm_ent_dict or load_embeddings_dict(lm_ent_path, embeddings_dim)
        self.lm_cc_dict = lm_cc_dict or load_embeddings_dict(lm_cc_path, embeddings_dim)
    
    def __getitem__(self, idx):
        d = self.sample_records[idx]
        _e, _cc, _lbl = d['neighbor'], d['concept'], d['label']
        emb_ent = self.emb_ent_dict[_e]
        emb_cc = self.emb_cc_dict[_cc]
        lm_ent = self.lm_ent_dict[_e]
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


class Wiki_EE_Dataset_2(Dataset):
    '''
    LM entity embeddings based on Hearst patterns, so it's concept-dependent
    '''
    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
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
        self.embeddings_dim = embeddings_dim
        
        self.sample_records = pd.read_csv(ds_path).to_dict('record')
        
        self.emb_ent_dict = emb_ent_dict or load_embeddings_dict(emb_ent_path, embeddings_dim)
        self.emb_cc_dict = emb_cc_dict or load_embeddings_dict(emb_cc_path, embeddings_dim)
        # self.lm_ent_dict = lm_ent_dict or load_embeddings_dict(lm_ent_path, embeddings_dim)
        if lm_ent_dict is not None:
            self.lm_ent_dict = lm_ent_dict
        else:
            lm_ent_df = pd.read_csv(lm_ent_path)
            self.lm_ent_dict = dict()
            for d in lm_ent_df.to_dict('record'):
                k = (d['concept'], d['neighbor'])
                v = [float(v) for v in d['embedding'].split(' ')]
                self.lm_ent_dict[k] = v
            
        self.lm_cc_dict = lm_cc_dict or load_embeddings_dict(lm_cc_path, embeddings_dim)
    
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
    


class Indeed_EE_Dataset(Dataset):
    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
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
        self.embeddings_dim = embeddings_dim
        
        self.emb_ent_dict = emb_ent_dict or load_embeddings_dict(emb_ent_path, embeddings_dim)
        self.emb_cc_dict = emb_cc_dict or load_embeddings_dict(emb_cc_path, embeddings_dim)
        self.lm_ent_dict = lm_ent_dict or load_embeddings_dict(lm_ent_path, embeddings_dim)
        self.lm_cc_dict = lm_cc_dict or load_embeddings_dict(lm_cc_path, embeddings_dim)
        
        gold_df = pd.read_csv(ds_path)
        _valid_ents = set(self.lm_ent_dict.keys()) & set(self.emb_ent_dict.keys())
        
        self.sample_records = []
        for d in gold_df.to_dict('record'):
            if d['neighbor'] not in _valid_ents:
#                 print(d['neighbor'], 'true but not valid??')
#                 print(d['neighbor'] in self.emb_ent_dict.keys(), d['neighbor'] in self.lm_ent_dict.keys())
#                 print()
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
        lm_ent = self.lm_ent_dict[_e]
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

    

class Indeed_EE_Dataset_2(Dataset):
    def __init__(self,
                 ds_path,
                 emb_ent_path, 
                 emb_cc_path, 
                 lm_ent_path, 
                 lm_cc_path,
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
        self.embeddings_dim = embeddings_dim
        
        self.emb_ent_dict = emb_ent_dict or load_embeddings_dict(emb_ent_path, embeddings_dim)
        self.emb_cc_dict = emb_cc_dict or load_embeddings_dict(emb_cc_path, embeddings_dim)
        # self.lm_ent_dict = lm_ent_dict or load_embeddings_dict(lm_ent_path, embeddings_dim)
        if lm_ent_dict is not None:
            self.lm_ent_dict = lm_ent_dict
        else:
            lm_ent_df = pd.read_csv(lm_ent_path)
            self.lm_ent_dict = dict()
            for d in lm_ent_df.to_dict('record'):
                k = (d['concept'], d['neighbor'])
                v = [float(v) for v in d['embedding'].split(' ')]
                self.lm_ent_dict[k] = v
            
        self.lm_cc_dict = lm_cc_dict or load_embeddings_dict(lm_cc_path, embeddings_dim)
        
        gold_df = pd.read_csv(ds_path)
        _valid_ents = set([e for cc, e in self.lm_ent_dict.keys()]) & set(self.emb_ent_dict.keys())
        
        self.sample_records = []
        for d in gold_df.to_dict('record'):
            if d['neighbor'] not in _valid_ents:
#                 print(d['neighbor'], 'true but not valid??')
#                 print(d['neighbor'] in self.emb_ent_dict.keys(), d['neighbor'] in self.lm_ent_dict.keys())
#                 print()
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

    
    