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
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils import bert_untokenize

from roberta_ses.interface import Roberta_SES_Entailment


class LMProbe(object):
    ''' Now supporting bert / roberta '''
    def __init__(self, model_name='bert-base-uncased', max_n_grams=5, use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.max_n_grams = max_n_grams

        self.mask_token = self.tokenizer.mask_token
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token

    def fill_multi_mask(self, input_txt, topk=3):
        print('Warning: LMProbe.fill_multi_mask() is not maintained')
        
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
        # input_txt: template sentence with 1 mask token 
        # cands: List[str], list of untokenized candidates with any length 
        cand_bins = {i : [] for i in range(1, self.max_n_grams + 1)}
        for c in cands:
            c_tokenized = self.tokenizer.tokenize(' ' + c)  ## prepending ' ' for roberta 
            if len(c_tokenized) > self.max_n_grams:
#                 print(f'{c_tokenized}: too many wordpieces')
                continue
            cand_bins[len(c_tokenized)].append(c_tokenized)
        
        all_cand_scores = []
        for c_len in range(1, self.max_n_grams + 1):
            _cands = cand_bins[c_len]
            if len(_cands) == 0:
                continue
            
            _input = self.cls_token + " " + \
                input_txt.replace("[MASK]",
                    self.mask_token + (" " + self.mask_token) * (c_len - 1)
                ) + " " + self.sep_token
                
            _cand_scores = self._score_ngram_candidates(_input, _cands)
            
            for d in _cand_scores:
                all_cand_scores.append({
                    'cand': self.tokenizer.convert_tokens_to_string(d['cand']).strip(),
                    'score': d['score']
                })
        
        return all_cand_scores
    
    def _score_ngram_candidates(self, input_txt, cands):
        # cands: List[List[str]], list of tokenized candidates 
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        
        if tokenized_txt[0] != self.cls_token or tokenized_txt[-1] != self.sep_token:
            raise Exception(f'Input string must start with {self.cls_token} and end with {self.sep_token}, got {tokenized_txt}')
        if self.mask_token not in tokenized_txt:
            raise Exception(f'Input string must have at least one {self.mask_token}, got {tokenized_txt}')
        
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == self.mask_token]
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
        
        self.tokenizer = self.bert_tokenizer
        self.max_n_grams = max_n_grams
        
    def score_candidates(self, input_txt, cands, renorm_n=5):
        # cands: List[str], list candidates (untokenzied) 
        
        if input_txt.count("[MASK]") != 1:
            raise Exception(f'Input string must have exactly one mask token, got {input_txt}')

        cand_bins = {i : [] for i in range(1, self.max_n_grams + 1)}
        for c in cands:
            c_tokenized = self.bert_tokenizer.tokenize(c)
            if len(c_tokenized) > self.max_n_grams:
#                 print(f'{c_tokenized}: too many wordpieces')
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
            _renorm_bert_scores = {bert_untokenize(d['cand']) : d['score'] for d in _renorm_cand_dicts}
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
            
            # returning scores are probabilities 
            _renormed_cand_scores = [
                {'cand': bert_untokenize(d['cand']),
                 'score': np.exp((d['score'] + _renorm_bias) / _gpt2_len)}
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
    
class LMProbe_PMI(object):
    def __init__(self, model_name='bert-base-uncased', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

    def fill_multi_mask(self, input_txt, topk=3):
        raise NotImplementedError
    
    def score_candidate_pair(self, input_txt, head, tail):
        # input_txt: str, with [HEAD] for head and [TAIL] for tail 
        # tail: str, the tail entity 
        # head: str, the head entity 
        # Return: PMI 
        
        head_toks = self.tokenizer.tokenize(head)
        head_len = len(head_toks)
        
        prob_with_head = self.score_tail(input_txt, tail, head=head)
        prob_wo_head = self.score_tail(input_txt, tail, head_len=head_len)
        pmi = np.log(prob_with_head) - np.log(prob_wo_head)
        return pmi
    
    def score_tail(self, input_txt, tail, head=None, head_len=None):
        # input_txt: str, with [HEAD] for head and [TAIL] for tail 
        # tail: str, the tail entity 
        # head: str, the head entity 
        # head_len: int, the length of head entity
        # Should only give head or head_len 
        
        assert (head is None) + (head_len is None) == 1, \
            f"head = {head}, head_len = {head_len}"
        assert input_txt.count("[HEAD]") == input_txt.count("[TAIL]") == 1, \
            f"Input string must have [HEAD] and [TAIL], got {input_txt}"
        head_first = (input_txt.index("[HEAD]") < input_txt.index("[TAIL]"))
        
        tail_toks = self.tokenizer.tokenize(tail)
        tail_len = len(tail_toks)
        input_txt = input_txt.replace('[TAIL]', '[MASK]' + ' [MASK]' * (tail_len-1))
        
        if head is not None:
            head_toks = self.tokenizer.tokenize(head)
            head_len = len(head_toks)
#             print(head_toks, head_len)
            input_txt = input_txt.replace('[HEAD]', head)
        else:
            input_txt = input_txt.replace('[HEAD]', '[MASK]' + ' [MASK]' * (head_len-1))
        
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        tokenized_txt = ['[CLS]'] + tokenized_txt + ['[SEP]']

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
        tokens_tensor = torch.tensor([indexed_tokens])
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]
        if head is not None:
            # head is not [MASK] 
            tail_indices = mask_indices
        elif head_first:
            # head is [MASK] and first 
            tail_indices = mask_indices[head_len:]
        else:
            # head is [MASK] and second 
            tail_indices = mask_indices[:tail_len]
#         print(tokenized_txt, tail_indices)
        
        segment_idx = tokens_tensor * 0
        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segment_idx.to(self.device)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        probs = torch.softmax(predictions, dim=-1)[0]
        probs = probs.detach().cpu().numpy()
        
        _scores = []
        tail_tok_ids = self.tokenizer.convert_tokens_to_ids(tail_toks)
        for i, token_id in zip(tail_indices, tail_tok_ids):
            _scores.append(probs[i, token_id].item())
        score = gmean(_scores)

        return score

    
class LMProbe_PMI_greedy(object):
    def __init__(self, model_name='bert-base-uncased', use_gpu=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.mask_token = self.tokenizer.mask_token

    def fill_multi_mask(self, input_txt, topk=3):
        raise NotImplementedError
        
    def score_candidate_pair(self, input_txt, head, tail):
        # input_txt: str, with [HEAD] for head and [TAIL] for tail 
        # tail: str, the tail entity 
        # head: str, the head entity 
        # Return: PMI 
        
        head_toks = self.tokenizer.tokenize(head)
        head_len = len(head_toks)
        
        prob_with_head = self.score_tail(input_txt, tail, head=head)
        prob_wo_head = self.score_tail(input_txt, tail, head_len=head_len)
        pmi = np.log(prob_with_head) - np.log(prob_wo_head)
        return pmi
    
    def score_tail(self, input_txt, tail, head=None, head_len=None):
        # input_txt: str, with [HEAD] for head and [TAIL] for tail 
        # tail: str, the tail entity 
        # head: str, the head entity 
        # head_len: int, the length of head entity
        # Should only give head or head_len 
        # head_first: bool, whether the first [MASK] is the head 
        
        assert (head is None) + (head_len is None) == 1, \
            f"head = {head}, head_len = {head_len}"
        assert input_txt.count("[HEAD]") == input_txt.count("[TAIL]") == 1, \
            f"Input string must have [HEAD] and [TAIL], got {input_txt}"
        head_first = (input_txt.index("[HEAD]") < input_txt.index("[TAIL]"))
                
        tail_toks = self.tokenizer.tokenize(tail)
        tail_len = len(tail_toks)
        input_txt = input_txt.replace('[TAIL]', '[MASK]' + ' [MASK]' * (tail_len-1))
        
        if head is not None:
            head_toks = self.tokenizer.tokenize(head)
            head_len = len(head_toks)
#             print(head_toks, head_len)
            input_txt = input_txt.replace('[HEAD]', head)
        else:
            input_txt = input_txt.replace('[HEAD]', '[MASK]' + ' [MASK]' * (head_len-1))
        
        tokenized_txt = self.tokenizer.tokenize(input_txt)
        tokenized_txt = ['[CLS]'] + tokenized_txt + ['[SEP]']
        mask_indices = [i for i, x in enumerate(tokenized_txt) if x == "[MASK]"]

        if head is not None:
            # head is not [MASK] 
            tail_indices = mask_indices
        elif head_first:
            # head is [MASK] and first 
            tail_indices = mask_indices[head_len:]
        else:
            # head is [MASK] and second 
            tail_indices = mask_indices[:tail_len]
        
        # Greedy filling 
        unfilled_indices = list(tail_indices)
        scores = []
        while len(unfilled_indices) > 0:
#             print(tokenized_txt, unfilled_indices)
            
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_txt)
            tokens_tensor = torch.tensor([indexed_tokens])
            
            segment_idx = tokens_tensor * 0
            tokens_tensor = tokens_tensor.to(self.device)
            segments_tensors = segment_idx.to(self.device)

            with torch.no_grad():
                outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
                predictions = outputs[0]

            probs = torch.softmax(predictions, dim=-1)[0]
            probs = probs.detach().cpu().numpy()

            _tok_scores = []
            tail_tok_ids = self.tokenizer.convert_tokens_to_ids(tail_toks)
            for i, token_id in zip(tail_indices, tail_tok_ids):
                if i not in unfilled_indices:
                    continue
                _score = probs[i, token_id].item()
                _tok_scores.append((i, token_id, _score))
                
            _tok_scores.sort(key=lambda p : p[1], reverse=True)
            _fill_idx, _fill_tok, _score = _tok_scores[0]
            unfilled_indices.remove(_fill_idx)
            scores.append(_score)
            tokenized_txt[_fill_idx] = self.tokenizer.convert_ids_to_tokens(_fill_tok)
        
        return np.prod(scores)
    
    
def main():
    pass
    
    
if __name__ == "__main__":
    main()