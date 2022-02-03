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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, 
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-o', '--test_output', type=str, 
                        required=True, help='Test output path')
    parser.add_argument('-ep', '--epochs', type=int, 
                        required=True, help='number of epochs')
#     parser.add_argument('-ename', '--embedding_name', type=str, default=None, required=False,
#                         help='Name of the embedding, to append to output file name. (optional)')
#     parser.add_argument('-m', '--model_path', type=str, required=True, default='bert-base-uncased',
#                         help='model_path')
#     parser.add_argument('-c', '--max_context_ct', type=int, default=500, help='max. no. of context to consider')
    args = parser.parse_args()
    return args


class EE_Classifier(pl.LightningModule):
    def __init__(self, 
#                  ent_paths,
#                  cc_paths,
                 embeddings_dim,
                 clf_type='mlp',
                 loss_type='xent', # 'xent' = cross entropy, 'rank' = margin ranking loss 
                 ranking_loss_margin=None,
                 extra_feats=['mult', 'sub'],
                 ff_dims=[256, 64],
                 diff_loss_margin=0.2,
                 emb_loss_coef=1.0,  # mostly for debugging purpose
                 lm_loss_coef=1.0,   # mostly for debugging purpose
                 joint_loss_coef=1.0,
                 diff_loss_coef=1.0,
                 optim_type='adam',
                 init_lr=1e-3):
        super().__init__()
        # ent_paths/cc_paths: Dict[view(emb/lm), path]
        
#         self.ent_paths = ent_paths
#         self.cc_paths = cc_paths
#         self.views = list(ent_paths.keys())
        self.views = ['emb', 'lm']  # Hard-coded for now 
        self.embeddings_dim = embeddings_dim
        self.clf_type = clf_type
        self.loss_type = loss_type
        self.ranking_loss_margin = ranking_loss_margin
        
        self.extra_feats = extra_feats
        self.input_dim = (2 + len(extra_feats)) * embeddings_dim
        self.ff_dims = ff_dims
        self.diff_loss_margin = diff_loss_margin
        self.emb_loss_coef = emb_loss_coef
        self.lm_loss_coef = lm_loss_coef
        self.joint_loss_coef = joint_loss_coef
        self.diff_loss_coef = diff_loss_coef
        
        self.optim_type = optim_type
        self.init_lr = init_lr

#         self.emb_ent_dict = load_embeddings_dict(emb_ent_path, embeddings_dim)
#         self.emb_cc_dict = load_embeddings_dict(emb_cc_path, embeddings_dim)
#         self.lm_ent_dict = load_embeddings_dict(lm_ent_path, embeddings_dim)
#         self.lm_cc_dict = load_embeddings_dict(lm_cc_path, embeddings_dim)

        if clf_type != 'mlp':
            raise NotImplementedError()

        self.views_clf_heads = nn.ModuleDict()
        for _v in self.views:
            _layers = [nn.Linear(self.input_dim, ff_dims[0]), nn.LeakyReLU(negative_slope=0.01)]
            for i in range(1, len(ff_dims)):
                _layers.append(nn.Linear(ff_dims[i-1], ff_dims[i]))
                _layers.append(nn.LeakyReLU(negative_slope=0.01))
            _layers.append(nn.Linear(ff_dims[-1], 1))
            _layers.append(nn.Sigmoid())
            
            _head = nn.Sequential(*_layers)
            self.views_clf_heads[_v] = _head

    def forward(self, in_batch):
        if self.loss_type == 'rank':
            return self._forward_pair(in_batch)
        else:
            return self._forward_single(in_batch)
    
    # New: ranking loss
    def _forward_pair(self, in_batch):
        pos_batch = {
            'emb_ent': in_batch['emb_pos_ent'],
            'emb_cc': in_batch['emb_pos_cc'],
            'lm_ent': in_batch['lm_pos_ent'],
            'lm_cc': in_batch['lm_pos_cc'],
        }
        
        neg_batch = {
            'emb_ent': in_batch['emb_neg_ent'],
            'emb_cc': in_batch['emb_neg_cc'],
            'lm_ent': in_batch['lm_neg_ent'],
            'lm_cc': in_batch['lm_neg_cc'],
        }
        
        pos_logits = self._forward_single(pos_batch)
        neg_logits = self._forward_single(neg_batch)
        return {
            'pos': pos_logits,
            'neg': neg_logits
        }
    
    def _forward_single(self, in_batch):
        v2ents = dict([(_v, in_batch[f'{_v}_ent']) for _v in self.views])
        v2ccs = dict([(_v, in_batch[f'{_v}_cc']) for _v in self.views])
        
        v2logits = dict()
        for _v in self.views:
            # (batch, emb_dim)
            ent = v2ents[_v]
            cc = v2ccs[_v]
            
            input_feats_list = [ent, cc]
            for _ef in self.extra_feats:
                if _ef == 'mult':
                    input_feats_list.append(ent * cc)
                elif _ef == 'sub':
                    input_feats_list.append(ent - cc)
                else:
                    raise ValueError(_ef)
            # (batch, input_dim)
            input_feats = torch.cat(input_feats_list, dim=-1)
            
            logits = self.views_clf_heads[_v](input_feats)
            v2logits[_v] = logits
        
        # Hardcoded loss for 2 views 
        emb_logits = v2logits['emb']
        lm_logits = v2logits['lm']
        joint_logits = torch.mean(torch.cat([emb_logits, lm_logits], dim=-1), dim=-1)
        
        return {
            'emb_logits': emb_logits.squeeze(-1),
            'lm_logits': lm_logits.squeeze(-1),
            'joint_logits': joint_logits,
        }
    
    def _compute_xent_loss(self, batch, batch_idx):
        pred_logits = self(batch)
        emb_logits = pred_logits['emb_logits']
        lm_logits = pred_logits['lm_logits']
        joint_logits = pred_logits['joint_logits']
        labels = batch['label'].to(torch.float32)
        
        emb_loss = F.binary_cross_entropy(emb_logits, labels)
        lm_loss = F.binary_cross_entropy(lm_logits, labels)
        joint_loss = F.binary_cross_entropy(joint_logits, labels)
        diff_loss = torch.linalg.norm(torch.clamp_min(torch.abs(emb_logits - lm_logits) - self.diff_loss_margin, 0))
        
        return {
            'emb_loss': emb_loss,
            'lm_loss': lm_loss,
            'joint_loss': joint_loss,
            'diff_loss': diff_loss,
        }
    
    # New: ranking loss
    def _compute_rank_loss(self, batch, batch_idx):
        pred_logits = self(batch)
        pos_emb_logits = pred_logits['pos']['emb_logits']
        pos_lm_logits = pred_logits['pos']['lm_logits']
        pos_joint_logits = pred_logits['pos']['joint_logits']
        neg_emb_logits = pred_logits['neg']['emb_logits']
        neg_lm_logits = pred_logits['neg']['lm_logits']
        neg_joint_logits = pred_logits['neg']['joint_logits']
        
        emb_loss = F.margin_ranking_loss(pos_emb_logits, neg_emb_logits,
                                         target=torch.ones_like(pos_emb_logits),
                                         margin=self.ranking_loss_margin)
        lm_loss = F.margin_ranking_loss(pos_lm_logits, neg_lm_logits,
                                        target=torch.ones_like(pos_lm_logits),
                                        margin=self.ranking_loss_margin)
        joint_loss = F.margin_ranking_loss(pos_joint_logits, neg_joint_logits,
                                           target=torch.ones_like(pos_joint_logits),
                                           margin=self.ranking_loss_margin)
        diff_loss = \
            torch.linalg.norm(torch.clamp_min(torch.abs(pos_emb_logits - pos_lm_logits) - self.diff_loss_margin, 0)) + \
            torch.linalg.norm(torch.clamp_min(torch.abs(neg_emb_logits - neg_lm_logits) - self.diff_loss_margin, 0))
        
        return {
            'emb_loss': emb_loss,
            'lm_loss': lm_loss,
            'joint_loss': joint_loss,
            'diff_loss': diff_loss,
        }
    
    def training_step(self, batch, batch_idx):
        if self.loss_type == 'rank':
            loss_dict = self._compute_rank_loss(batch, batch_idx)
        elif self.loss_type == 'xent':
            loss_dict = self._compute_xent_loss(batch, batch_idx)
        else:
            raise ValueError(self.loss_type)
        
        emb_loss = loss_dict['emb_loss']
        lm_loss = loss_dict['lm_loss']
        joint_loss = loss_dict['joint_loss']
        diff_loss = loss_dict['diff_loss']
        
        loss = self.emb_loss_coef * emb_loss + self.lm_loss_coef * lm_loss + \
            self.joint_loss_coef * joint_loss + self.diff_loss_coef * diff_loss
        
        self.log("emb_loss", emb_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lm_loss", lm_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("joint_loss", joint_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("diff_loss", diff_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            if self.loss_type == 'rank':
                loss_dict = self._compute_rank_loss(batch, batch_idx)
            elif self.loss_type == 'xent':
                loss_dict = self._compute_xent_loss(batch, batch_idx)
            else:
                raise ValueError(self.loss_type)
        
        emb_loss = loss_dict['emb_loss']
        lm_loss = loss_dict['lm_loss']
        joint_loss = loss_dict['joint_loss']
        diff_loss = loss_dict['diff_loss']
        
        loss = self.emb_loss_coef * emb_loss + self.lm_loss_coef * lm_loss + \
            self.joint_loss_coef * joint_loss + self.diff_loss_coef * diff_loss
        
        self.log("emb_loss", emb_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("lm_loss", lm_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("joint_loss", joint_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("diff_loss", diff_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            pred_logits = self._forward_single(batch)
            
        emb_logits = pred_logits['emb_logits'].detach().cpu().numpy()
        lm_logits = pred_logits['lm_logits'].detach().cpu().numpy()
        joint_logits = pred_logits['joint_logits'].detach().cpu().numpy()
        labels = batch['label'].detach().cpu().numpy()
        
        emb_pred = (emb_logits > 0.5).astype(int)
        lm_pred = (lm_logits > 0.5).astype(int)
        joint_pred = (joint_logits > 0.5).astype(int)
        
        res_dict = {
            'emb_logits': emb_logits,
            'lm_logits': lm_logits,
            'joint_logits': joint_logits,
            'emb_pred': emb_pred,
            'lm_pred': lm_pred,
            'joint_pred': joint_pred,
            'label': labels
        }
        return res_dict
    
    def configure_optimizers(self):
        if self.optim_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        elif self.optim_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.init_lr)
        else:
            raise NotImplementedError(self.optim_type)
            
        return optimizer

    
    
    