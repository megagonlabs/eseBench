from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import os
import torch
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel, AutoConfig

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations

from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
spacy_tokenizer = nlp.tokenizer

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
    parser.add_argument('-rd', '--raw_dataset_dir', type=str,
                        required=True, help='Dataset path to raw dataset') # '/home/ubuntu/users/nikita/data/indeed/indeedQA'
    args = parser.parse_args()
    return args


def build_corpus(dataset_path,
                 benchmark_path,
                 embed_num_path,
                 raw_dataset_path,
                 raw_company_path,
                 out_corpus_path,
                 **kwargs):
    
    with open(embed_num_path, 'r') as f:
        entities = [l.strip().rsplit(' ', 1)[0] for l in f.readlines()]

    df_dataset = pd.read_csv(raw_dataset_path) 
    df_dataset = df_dataset[df_dataset['answerContent'].notna()]
    df_company = pd.read_csv(raw_company_path)

    df_merged_dataset = df_dataset.merge(df_company, how='inner', on='fccompanyId')
                 
    out_corpus = []

    for i, row in tqdm(df_merged_dataset.iterrows(), total=df_merged_dataset.shape[0], desc='Processing lines'):
        company = row["companyName"]
        ans = row["answerContent"]
        ans_nlp = nlp(ans)
        for sent in ans_nlp.sents:
            sent_tok_list = [str(t) for t in sent]
            _s = f' {" ".join(sent_tok_list)} '.lower()
            _ents = []
            if company.lower() in entities:
                _ents.append(company.lower())
            for _e in entities:
                if f' {_e} ' in _s:
                    _ents.append(_e)
            _ents = list(set(_ents))
            
            out_corpus.append({
                "tokens": sent_tok_list,
                "company": company,
                "entities": _ents,
            })
            
    with open(out_corpus_path, 'w') as f:
        for d in out_corpus:
            f.write(json.dumps(d) + '\n')
    

def main():
    args = parse_arguments()
    args.embed_num_path = os.path.join(args.dataset_path, 'BERTembednum+seeds.txt')
    args.raw_dataset_path = os.path.join(args.raw_dataset_dir, 'question_answers.csv')
    args.raw_company_path = os.path.join(args.raw_dataset_dir, 'fccid-companyName.csv')
    args.out_corpus_path = os.path.join(args.dataset_path, 'sentences_with_company-test.json')

    build_corpus(**vars(args))

if __name__ == "__main__":
    main()


