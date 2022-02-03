from tqdm import tqdm
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

import pandas as pd
import os
import sys
import math
from annoy import AnnoyIndex
import matplotlib
from matplotlib import pyplot as plt
import networkx as nx
from glob import glob


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus_path', type=str, default=None,
                        help='Corpus path (.../source/corpus.txt)')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='Batch size (if None, not batching)')

    args = parser.parse_args()
    return args

def compute_gpt2_perplexity(corpus_path, batch_size):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    with open(corpus_path, 'r') as f:
        corpus_lines = [l.strip() for l in f if len(l.strip()) > 0]
    print('Number of lines:', len(corpus_lines))
    
    total_ppl = 0
    total_toks = 0
    
    if batch_size is None:
        ## Unbatched 
        for l in tqdm(corpus_lines):
            l_tokenized = gpt2_tokenizer.encode_plus(l, return_tensors='pt', truncation=True)
            num_toks = l_tokenized['attention_mask'].sum() - 1
            l_ppl = gpt2_model(**l_tokenized, labels=l_tokenized["input_ids"])[0].item()
            total_ppl += l_ppl * num_toks
            total_toks += num_toks
        
    else:
        ## Batched 
        corpus_lines.sort(key=lambda l: -len(l))
        
        for i in tqdm(range(0, len(corpus_lines), batch_size)):
            _lines = corpus_lines[i : i + batch_size]
            _eff_batch_size = len(_lines)

            _lines_tokenized = gpt2_tokenizer.batch_encode_plus(
                _lines, return_tensors='pt', padding=True, truncation=True)
            _ppl = gpt2_model(**_lines_tokenized, labels=_lines_tokenized["input_ids"])[0].item()
            _num_toks = _lines_tokenized['attention_mask'].sum() - _eff_batch_size  ## -1 for each line 

            total_ppl += _ppl * _num_toks
            total_toks += _num_toks
    
    
    tok_avg_ppl = total_ppl / total_toks
    
    print(f'Tokens = {total_toks}')
    print(f'Total ppl = {total_ppl:.4f}')
    print(f'Token average ppl = {tok_avg_ppl:.4f}')
    

def main():
    args = parse_arguments()

    random.seed(123)
    compute_gpt2_perplexity(**vars(args))


if __name__ == "__main__":
    main()