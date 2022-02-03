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
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback

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

# from multiview_EE_datasets import Wiki_EE_Dataset, Wiki_EE_Dataset_2, \
#     Wiki_EE_Dataset_2_pairs, Wiki_EE_Dataset_2_iter_pairs, \
#     Indeed_EE_Dataset, Indeed_EE_Dataset_2
from multiview_EE_datasets_NEW import EE_Dataset, EE_Dataset_pairs
from multiview_EE_models import EE_Classifier


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str, 
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-dn', '--dataset_name', type=str, 
                        required=True, help='Dataset name (wiki / indeed)')
    parser.add_argument('-o_dir', '--test_output_dir', type=str, 
                        required=True, help='Test output path')
    parser.add_argument('-loss', '--loss_type', type=str, choices=['xent', 'rank'],
                        required=True, help='Loss type')
    parser.add_argument('-r_margin', '--ranking_loss_margin', type=float, default=0,
                        required=False, help='Ranking loss margin')
    parser.add_argument('-rsmp', '--resampling', action='store_true',
                        help='Resampling pos/neg pairs every epoch')
    parser.add_argument('-ep', '--epochs', type=int, default=1000,
                        required=False, help='number of epochs')
    parser.add_argument('-v', '--version', type=str, default=None,
                        required=False, help='version number')
    parser.add_argument('-g_clip', '--gradient_clip_val', type=float, default=0.0,
                        required=False, help='gradient clipping value, 0.0 = no clipping')
    parser.add_argument('-w_emb', '--emb_loss_coef', type=float, default=1.0,
                        required=False, help='weight of embedding head loss')
    parser.add_argument('-w_lm', '--lm_loss_coef', type=float, default=1.0,
                        required=False, help='weight of lm head loss')
    parser.add_argument('-w_joint', '--joint_loss_coef', type=float, default=1.0,
                        required=False, help='weight of joint head loss')
    parser.add_argument('-w_diff', '--diff_loss_coef', type=float, default=1.0,
                        required=False, help='weight of diff loss')
    parser.add_argument('-hearst', '--lm_ent_hearst', action='store_true',
                        help='using hearst-based lm_ent embedding, instead of input embedding')
    parser.add_argument('--test_only', action='store_true',
                        help='only do testing')
    parser.add_argument('-ckpt', '--trained_model_ckpt', type=str, default=None, 
                        required=False, help='the checkpoint used for testing only')
#     parser.add_argument('-ename', '--embedding_name', type=str, default=None, required=False,
#                         help='Name of the embedding, to append to output file name. (optional)')
#     parser.add_argument('-m', '--model_path', type=str, required=True, default='bert-base-uncased',
#                         help='model_path')
#     parser.add_argument('-c', '--max_context_ct', type=int, default=500, help='max. no. of context to consider')
    args = parser.parse_args()
    return args

    
def main():
    args = parse_arguments()
    
    data_dir = args.dataset_path
    train_data_path = os.path.join(data_dir, f'{args.dataset_name}_ee_train.csv')
    test_in_cc_data_path = os.path.join(data_dir, f'{args.dataset_name}_ee_test_in_concept.csv')
    test_in_dom_data_path = os.path.join(data_dir, f'{args.dataset_name}_ee_test_in_domain.csv')
    
    if args.dataset_name == 'wiki':
        emb_ent_path = os.path.join(data_dir, 'BERTembed_gt.txt')
        emb_cc_path = os.path.join(data_dir, 'BERTembed_gt_concepts.txt')
        lm_cc_path = os.path.join(data_dir, 'BERTembed_gt_lm_concepts.txt')
        lm_ent_path = os.path.join(data_dir, 'BERTembed_gt_lm_entities_hearst.csv')
    elif args.dataset_name == 'indeed':
        emb_ent_path = os.path.join(data_dir, 'BERTembed+seeds.txt')
        emb_cc_path = os.path.join(data_dir, 'BERTembed_concepts.txt')
        lm_cc_path = os.path.join(data_dir, 'BERTembed_lm_concepts.txt')
        lm_ent_path = os.path.join(data_dir, 'BERTembed_lm_entities_hearst.csv')
    else:
        # Yelp or TripAdvisor 
        emb_ent_path = os.path.join(data_dir, 'BERTembed.txt')
        emb_cc_path = os.path.join(data_dir, 'BERTembed_concepts.txt')
        lm_cc_path = os.path.join(data_dir, 'BERTembed_lm_concepts.txt')
        lm_ent_path = os.path.join(data_dir, 'BERTembed_lm_entities_hearst.csv')
    
    
    if args.lm_ent_hearst:
        train_dataset_cls = EE_Dataset if args.loss_type == 'xent' else EE_Dataset_pairs
        dev_dataset_cls = EE_Dataset if args.loss_type == 'xent' else EE_Dataset_pairs
        test_dataset_cls = EE_Dataset
    else:
        ## old version, unstable, no longer used 
        assert False
#         dataset_cls = Wiki_EE_Dataset
#         lm_ent_path = os.path.join(wiki_data_dir, 'BERTembed_gt_lm_entities.txt')
    
    
    print('Loading datasets...')
    train_set = train_dataset_cls(
        lm_ent_using_hearst=True,
        labels_in_datafile=True,
        ds_path=train_data_path, 
        emb_ent_path=emb_ent_path, 
        emb_cc_path=emb_cc_path, 
        lm_ent_path=lm_ent_path, 
        lm_cc_path=lm_cc_path,
    )
    # when using iter datasets, shuffle = False
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    
    dev_set = dev_dataset_cls(
        lm_ent_using_hearst=True,
        labels_in_datafile=True,
        ds_path=test_in_cc_data_path, 
        emb_ent_path=emb_ent_path,
        emb_cc_path=emb_cc_path,
        lm_ent_path=lm_ent_path,
        lm_cc_path=lm_cc_path,
        emb_ent_dict=train_set.emb_ent_dict,
        emb_cc_dict=train_set.emb_cc_dict,
        lm_ent_dict=train_set.lm_ent_dict,
        lm_cc_dict=train_set.lm_cc_dict,
    )
    dev_loader = DataLoader(dev_set, batch_size=4, shuffle=False)
    
    test_in_concept_set = test_dataset_cls(
        lm_ent_using_hearst=True,
        labels_in_datafile=True,
        ds_path=test_in_cc_data_path,
        emb_ent_path=emb_ent_path,
        emb_cc_path=emb_cc_path,
        lm_ent_path=lm_ent_path,
        lm_cc_path=lm_cc_path,
        emb_ent_dict=train_set.emb_ent_dict,
        emb_cc_dict=train_set.emb_cc_dict,
        lm_ent_dict=train_set.lm_ent_dict,
        lm_cc_dict=train_set.lm_cc_dict,
    )
    test_in_concept_loader = DataLoader(test_in_concept_set, batch_size=1, shuffle=False)
    
    test_in_domain_set = test_dataset_cls(
        lm_ent_using_hearst=True,
        labels_in_datafile=True,
        ds_path=test_in_dom_data_path,
        emb_ent_path=emb_ent_path,
        emb_cc_path=emb_cc_path,
        lm_ent_path=lm_ent_path,
        lm_cc_path=lm_cc_path,
        emb_ent_dict=train_set.emb_ent_dict,
        emb_cc_dict=train_set.emb_cc_dict,
        lm_ent_dict=train_set.lm_ent_dict,
        lm_cc_dict=train_set.lm_cc_dict,
    )
    test_in_domain_loader = DataLoader(test_in_domain_set, batch_size=1, shuffle=False)
    print('Done loading datasets.')
    
    if args.test_only:
        trainer = pl.Trainer(gpus=[0])
        ee_clf = EE_Classifier.load_from_checkpoint(args.trained_model_ckpt, embeddings_dim=768)
    else:
        callbacks = []
        if args.loss_type == 'rank' and args.resampling:
            print('Add resampling callback')
            callbacks.append(LambdaCallback(on_train_epoch_start=lambda trainer, pl_module: \
                                pl_module.train_dataloader.dataloader.dataset._load_sample_pair_records()))
        
        trainer = pl.Trainer(gpus=[0],
                             max_epochs=args.epochs,
                             gradient_clip_val=args.gradient_clip_val,
                             logger=TensorBoardLogger(save_dir='lightning_logs', version=args.version),
                             callbacks=callbacks)
        ee_clf = EE_Classifier(embeddings_dim=768,
                               loss_type=args.loss_type,
                               ranking_loss_margin=args.ranking_loss_margin,
                               emb_loss_coef=args.emb_loss_coef,
                               lm_loss_coef=args.lm_loss_coef,
                               joint_loss_coef=args.joint_loss_coef,
                               diff_loss_coef=args.diff_loss_coef)
        trainer.fit(ee_clf,
                    train_dataloaders=train_loader,
                    val_dataloaders=dev_loader)

    # in-concept
    pred_results = trainer.predict(ee_clf, dataloaders=test_in_concept_loader)
#     emb_preds = [d['emb_pred'] for d in pred_results]
#     lm_preds = [d['lm_pred'] for d in pred_results]
#     joint_preds = [d['joint_pred'] for d in pred_results]
#     labels = [d['label'] for d in pred_results]
    
    test_records = []
    for i in range(len(test_in_concept_set)):
        _raw_d = test_in_concept_set.sample_records[i]
#         _r = dict(_raw_record) # concept, neighbor, label 
        _pred_d = pred_results[i]
        
        _r = {
            'concept': _raw_d['concept'],
            'neighbor': _raw_d['neighbor'],
            'label': _raw_d['label'],
            'emb_logits': _pred_d['emb_logits'][0],
            'lm_logits': _pred_d['lm_logits'][0],
            'joint_logits': _pred_d['joint_logits'][0],
            'emb_pred': _pred_d['emb_pred'][0],
            'lm_pred': _pred_d['lm_pred'][0],
            'joint_pred': _pred_d['joint_pred'][0],
        }
        test_records.append(_r)
    
    os.makedirs(args.test_output_dir, exist_ok=True)
#     test_out_path = os.path.join(args.test_output_dir, f'ee_clf_test_in_concept_v={args.version}.csv')
    test_out_path = os.path.join(args.test_output_dir,
        f'ee_cotraining-v={args.version}-test_in_concept.csv')
    test_df = pd.DataFrame(test_records)
    test_df.to_csv(test_out_path, index=None)
    
    print('='*10, 'In-concept test', '='*10)
    print('Number of test samples:', test_df.shape[0])
    print('Emb accuracy:', sum(test_df['emb_pred'] == test_df['label']) / test_df.shape[0])
    print('LM accuracy:', sum(test_df['lm_pred'] == test_df['label']) / test_df.shape[0])
    print('Joint accuracy:', sum(test_df['joint_pred'] == test_df['label']) / test_df.shape[0])
    
    # in-domain
    pred_results = trainer.predict(ee_clf, dataloaders=test_in_domain_loader)
#     emb_preds = [d['emb_pred'] for d in pred_results]
#     lm_preds = [d['lm_pred'] for d in pred_results]
#     joint_preds = [d['joint_pred'] for d in pred_results]
#     labels = [d['label'] for d in pred_results]
    
    test_records = []
    for i in range(len(test_in_domain_set)):
        _raw_d = test_in_domain_set.sample_records[i]
#         _r = dict(_raw_record) # concept, neighbor, label 
        _pred_d = pred_results[i]
        
        _r = {
            'concept': _raw_d['concept'],
            'neighbor': _raw_d['neighbor'],
            'label': _raw_d['label'],
            'emb_logits': _pred_d['emb_logits'][0],
            'lm_logits': _pred_d['lm_logits'][0],
            'joint_logits': _pred_d['joint_logits'][0],
            'emb_pred': _pred_d['emb_pred'][0],
            'lm_pred': _pred_d['lm_pred'][0],
            'joint_pred': _pred_d['joint_pred'][0],
        }
        test_records.append(_r)
        
#     test_out_path = os.path.join(args.test_output_dir, f'ee_clf_test_in_domain_v={args.version}.csv')
    test_out_path = os.path.join(args.test_output_dir,
        f'ee_cotraining-v={args.version}-test_in_domain.csv')
    test_df = pd.DataFrame(test_records)
    test_df.to_csv(test_out_path, index=None)
    
    print('='*10, 'In-domain test', '='*10)
    print('Number of test samples:', test_df.shape[0])
    print('Emb accuracy:', sum(test_df['emb_pred'] == test_df['label']) / test_df.shape[0])
    print('LM accuracy:', sum(test_df['lm_pred'] == test_df['label']) / test_df.shape[0])
    print('Joint accuracy:', sum(test_df['joint_pred'] == test_df['label']) / test_df.shape[0])
    
if __name__ == '__main__':
    main()


