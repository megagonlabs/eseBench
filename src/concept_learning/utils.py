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


def _bert_untokenize(pieces):
    return ' '.join(pieces).replace(' ##', '')


class LMProbe(object):
    def __init__(self, model_name='bert-base-uncased', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

    def fill_multi_mask(self, input_txt, topk=3):
        if not (input_txt.startswith('[CLS]') and input_txt.endswith('[SEP]')):
            raise Exception('Input string must start with [CLS] and end with [SEP]')
        if not '[MASK]' in input_txt:
            raise Exception('Input string must have at least one mask token')
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        sorted_probs = sorted_probs.detach().cpu().numpy()
        sorted_idx = sorted_idx.detach().cpu().numpy()

        masked_cands = []
        for k in range(topk):
            predicted_indices = [sorted_idx[i, k].item() for i in mask_indices]
            predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_indices)
            predicted_probs = [sorted_probs[i, k].item() for i in mask_indices]
            seq = []
            for token_id, token, prob, masked_index in zip(predicted_indices, predicted_tokens, predicted_probs,
                                                           mask_indices):
                seq.append({"token": token_id, "token_str": token, "prob": prob, "masked_pos": masked_index})
            masked_cands.append(seq)

        return masked_cands
    
    def score_candidates(self, input_txt, cands):
        # cands: List[List[str]], list of tokenized candidates 
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        
        if tokenized_txt[0] != "[CLS]" or tokenized_txt[-1] != "[SEP]":
            raise Exception(f'Input string must start with [CLS] and end with [SEP], got {input_txt}')
        if "[MASK]" not in tokenized_txt:
            raise Exception(f'Input string must have at least one mask token, got {input_txt}')
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        probs = probs.detach().cpu().numpy()

        cand_scores = []
        for c in cands:
            assert len(c) == len(mask_indices), f'cand {c}; len(mask_indices) = {len(mask_indices)}'

            _scores = []
            c_token_ids = self.tokenizer.convert_tokens_to_ids(c)
            for i, token_id in zip(mask_indices, c_token_ids):
                _scores.append(probs[i, token_id].item())
            score = gmean(_scores)
            cand_scores.append({"cand": c, "score": score})

        cand_scores.sort(key=lambda d : d["score"], reverse=True)
        return cand_scores
    
class LMProbe_GPT2(object):
    def __init__(self, model_name='gpt2', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, return_dict=True)
        self.model.to(self.device)
        self.model.eval()

    def fill_multi_mask(self, input_txt, topk=3):
        raise Error("Mask filling not yet implemented for LMProbe_GPT2")
    
    def score_candidates(self, input_txt, cands):
        # cands: List[str], list candidates (untokenzied) 
        
        if input_txt.count("[MASK]") != 1:
            raise Exception(f'Input string must have exactly one mask token, got {input_txt}')

        cand_scores = []
        for c in cands:
            cand_input_txt = input_txt.replace("[MASK]", c)
            tokenized_input = self.tokenizer(cand_input_txt, return_tensors="pt")
            with torch.no_grad():
                model_outputs = self.model(**tokenized_input, labels=tokenized_input["input_ids"])
            score = np.exp(-model_outputs.loss.item())
            cand_scores.append({"cand": c, "score": score})

        cand_scores.sort(key=lambda d : d["score"], reverse=True)
        return cand_scores
    

# LMProbe_Joint with bert probs renormalized by gpt2 
class LMProbe_Joint(object):
    def __init__(self,
                 bert_model_name='bert-base-uncased',
                 gpt2_model_name='gpt2',
                 max_n_grams=5,
                 use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertForMaskedLM.from_pretrained(bert_model_name)
        self.bert_model.to(self.device)
        self.bert_model.eval()
        self.bert_mask_token = self.bert_tokenizer.mask_token
        
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name, return_dict=True)
        self.gpt2_model.to(self.device)
        self.gpt2_model.eval()
        
        self.max_n_grams = max_n_grams
        
    def joint_score_candidates(self, input_txt, cands, renorm_n=10):
        # cands: List[str], list candidates (untokenzied) 
        
        if input_txt.count("[MASK]") != 1:
            raise Exception(f'Input string must have exactly one mask token, got {input_txt}')

        cand_bins = {i : [] for i in range(1, self.max_n_grams + 1)}
        for c in cands:
            c_tokenized = self.bert_tokenizer.tokenize(c)
            if len(c_tokenized) > self.max_n_grams:
                print(f'{c_tokenized}: too many wordpieces')
                continue
            cand_bins[len(c_tokenized)].append(c_tokenized)
        
        all_cand_scores = []
        for c_len in range(1, self.max_n_grams + 1):
            _cands = cand_bins[c_len]
            if len(_cands) == 0:
                continue
            
            _input = "[CLS] " + input_txt.replace("[MASK]", "[MASK]" + " [MASK]" * (c_len - 1)) + " [SEP]"
            _cand_scores = self.bert_score_candidates(_input, _cands)
            
            _renorm_cand_dicts = _cand_scores[:renorm_n]
            _renorm_bert_scores = {_bert_untokenize(d['cand']) : d['score'] for d in _renorm_cand_dicts}
            _renorm_cands = list(_renorm_bert_scores.keys())
            
            _gpt2_cand_scores = self.gpt2_score_candidates(input_txt, _renorm_cands)
            _renorm_gpt2_scores = {d['cand'] : d['score'] for d in _gpt2_cand_scores}
            
#             print('BERT scores:')
#             print(json.dumps(_renorm_bert_scores, indent=2))
#             print('GPT2 scores:')
#             print(json.dumps(_renorm_gpt2_scores, indent=2))
#             if len(_renorm_cands) > 2:
#                 print('Pearson:')
#                 print(pearsonr(
#                     np.exp([_renorm_bert_scores[c] for c in _renorm_cands]),
#                     np.exp([_renorm_gpt2_scores[c] for c in _renorm_cands])))
            
            # bert_ll + _renorm_bias -> gpt2_ll
            _renorm_bias = np.log(np.sum(np.exp(list(_renorm_gpt2_scores.values())))) \
                - np.log(np.sum(np.exp(list(_renorm_bert_scores.values()))))
            
            _gpt2_len = len(self.gpt2_tokenizer(input_txt.replace('[MASK]', _renorm_cands[0]))['input_ids'])
            
            _renormed_cand_scores = [
                {'cand': _bert_untokenize(d['cand']),
                 'score': (d['score'] + _renorm_bias) / _gpt2_len}
                for d in _cand_scores
            ]
            all_cand_scores.extend(_renormed_cand_scores)
        
        all_cand_scores.sort(key=lambda d : d['score'], reverse=True)
        return all_cand_scores
    
    
    def bert_score_candidates(self, input_txt, cands):
        # cands: List[List[str]], list of tokenized candidates 
        tokenized_txt = self.bert_tokenizer.tokenize(input_txt)
        
        if tokenized_txt[0] != "[CLS]" or tokenized_txt[-1] != "[SEP]":
            raise Exception(f'Input string must start with [CLS] and end with [SEP], got {input_txt}')
        if "[MASK]" not in tokenized_txt:
            raise Exception(f'Input string must have at least one mask token, got {input_txt}')
        
        indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        probs = probs.detach().cpu().numpy()

        cand_scores = []
        for c in cands:
            assert len(c) == len(mask_indices), f'cand {c}; len(mask_indices) = {len(mask_indices)}'

            _scores = []
            c_token_ids = self.bert_tokenizer.convert_tokens_to_ids(c)
            for i, token_id in zip(mask_indices, c_token_ids):
                _scores.append(probs[i, token_id].item())
            score = np.sum(np.log(_scores))  # sum(log(p))
            cand_scores.append({"cand": c, "score": score})

        cand_scores.sort(key=lambda d : d["score"], reverse=True)
        return cand_scores
    
    def gpt2_score_candidates(self, input_txt, cands):
        # cands: List[str], list candidates (untokenzied) 
        
        if input_txt.count("[MASK]") != 1:
            raise Exception(f'Input string must have exactly one mask token, got {input_txt}')

        cand_scores = []
        for c in cands:
            cand_input_txt = input_txt.replace("[MASK]", c)
            tokenized_input = self.gpt2_tokenizer(cand_input_txt, return_tensors="pt")
            with torch.no_grad():
                model_outputs = self.gpt2_model(**tokenized_input, labels=tokenized_input["input_ids"])
                
            _input_len = tokenized_input['input_ids'].size(1)
            score = -model_outputs.loss.item() * (_input_len - 1)  # (log(p))
            cand_scores.append({"cand": c, "score": score})

        cand_scores.sort(key=lambda d : d["score"], reverse=True)
        return cand_scores
    
    
def main():
    pass
    
    
if __name__ == "__main__":
    main()