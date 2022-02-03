from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import pandas as pd

from utils import load_embeddings, load_seed_aligned_concepts, load_seed_aligned_relations, load_benchmark
from utils import load_EE_labels
from utils import get_masked_contexts, bert_untokenize
from utils import learn_patterns

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-scp', '--seed_concepts_path', type=str, required=True, help='seed_concept_path')
    parser.add_argument('-enp', '--embed_num_path', type=str, required=True, help='embed_num_path')
    parser.add_argument('-es', '--emb_src', type=str, required=True, help='emb_src (pre-computed BERT emb)')
    parser.add_argument('-m', '--model_path', type=str, required=False, default='bert-base-uncased',
                        help='model_path')
    parser.add_argument('-ename', '--embedding_name', type=str, required=False, default=None,
                        help='embedding name in output file name')
    parser.add_argument('--lm_ent_hearst', action='store_true',
                        help='Using Hearst for LM ent embeddings')
#     parser.add_argument('-c', '--max_context_ct', type=int, default=500, help='max. no. of context to consider')
#     parser.add_argument('-o', '--dest', type=str, required=False, default=None,
#                         help='Path to output file')
    args = parser.parse_args()
    return args

def ensure_tensor_on_device(device, **inputs):
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0].cpu() # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().cpu()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_vector(hidden_layers, token_index=0, mode='concat', top_n_layers=4):
    if mode == 'concat':
        out = torch.cat(tuple(hidden_layers[-top_n_layers:]), dim=-1)
        if len(out[token_index]) != 3072:
            print('shouldn"t happen')
        # print('output', out.size())
        # print(out[token_index].size())
        return out[token_index]

    if mode == 'average':
        # avg last 4 layer outputs
        return torch.stack(hidden_layers[-top_n_layers:]).mean(0)[token_index]

    if mode == 'sum':
        # sum last 4 layer outputs
        return torch.stack(hidden_layers[-top_n_layers:]).sum(0)[token_index]

    if mode == 'last':
        # last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers[-1:][0][token_index]

    if mode == 'second_last':
        # last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers[-2:-1][0][token_index]
    return None


def get_entity_span(context_ids, entity_ids):
    l = len(entity_ids)
    for i in range(len(context_ids)):
        if context_ids[i:i + l] == entity_ids:
            return i, i+l
    return []


def get_lm_probe_concept_embeddings(model_path, seed_concepts_path, lm_cc_dest=None, **kwargs):
    '''
    LM probe concept embeddings, using mask embeddings in Hearst pattern prompts
    :param model_path:
    :param seed_concepts_path:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mask_token_id = tokenizer.mask_token_id
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_instances_dict = dict(zip(
        seed_concepts_df['alignedCategoryName'].tolist(),
        seed_concepts_df['seedInstances'].tolist()
    ))
    
    probe_prompts = [
        "{0}, such as {1}, {2} and {3}.",
        "{0}, including {1}, {2} and {3}.",
        "{1}, {2}, {3} and other {0}.",
    ]

    concept_embeddings = {}
    for cc, seeds in tqdm(seed_instances_dict.items(), total=len(seed_instances_dict), desc="lm_probe concept embeddings"):
        cc_phrase = ' '.join(cc.split('_'))
        filled_prompts = [pr.format(cc_phrase, ', '.join(seeds[:-1]), tokenizer.mask_token, seeds[-1]) for pr in probe_prompts]
        
        encoded_input = tokenizer.batch_encode_plus(filled_prompts, return_token_type_ids=True, add_special_tokens=True, max_length=128, return_tensors='pt', padding=True, pad_to_max_length=True, truncation=True)

        mask = encoded_input['input_ids'] == mask_token_id  ## YS: only use the [MASK] embedding  
        with torch.no_grad():
            encoded_input = ensure_tensor_on_device(device, **encoded_input)
            model_output = model(**encoded_input)  # Compute token embeddings
                          
        cc_embeddings = mean_pooling(model_output, mask)  # mean pooling
        cc_embedding = torch.mean(cc_embeddings, dim=0).cpu().detach().numpy().tolist()
        concept_embeddings[cc] = cc_embedding
        
    print('Saving lm concept embeddings')
    if lm_cc_dest is not None:
        with open(lm_cc_dest, 'w') as f:
            for _cc, _emb in concept_embeddings.items():
                f.write("{} {}\n".format(_cc, ' '.join([str(x) for x in _emb])))
        
    return concept_embeddings

def get_lm_probe_entity_embeddings(model_path, embed_num_path, lm_ent_dest=None, **kwargs):
    '''
    LM probe entity embeddings, using BERT MLM output embeddings 
    :param model_path:
    :param embed_num_path:
    :return:
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    
    with open(embed_num_path, 'r') as f:
        entity_list = [l.strip().rsplit(' ', 1)[0] for l in f]

    # Seems that BERT uses same input embedding as MLM output embedding   
    emb_weights = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    
    entity_embeddings = {}
    for _e in tqdm(entity_list, desc="lm_probe entity embeddings"):
        _e_tok_ids = tokenizer.encode(_e, add_special_tokens=False)
        _emb = emb_weights[_e_tok_ids].mean(axis=0)
        entity_embeddings[_e] = _emb
    
    print('Saving lm entity embeddings')
    if lm_ent_dest is not None:
        with open(lm_ent_dest, 'w') as f:
            for _e, _emb in entity_embeddings.items():
                f.write("{} {}\n".format(_e, ' '.join([str(x) for x in _emb])))
    
    return entity_embeddings
    
    
def get_lm_probe_entity_embeddings_hearst(model_path, seed_concepts_path, embed_num_path, lm_ent_dest=None, **kwargs):
    '''
    LM probe entity embeddings with Hearst patterns, using embeddings of entity (filling mask) in Hearst patterns 
    :param model_path:
    :param seed_concepts_path:
    :param embed_num_path:
    :return:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_instances_dict = dict(zip(
        seed_concepts_df['alignedCategoryName'].tolist(),
        seed_concepts_df['seedInstances'].tolist()
    ))
    
    probe_prompts = [
        "{0}, such as {1}, {2} and {3}.",
        "{0}, including {1}, {2} and {3}.",
        "{1}, {2}, {3} and other {0}.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.to(device)
    
    with open(embed_num_path, 'r') as f:
        entity_list = [l.strip().rsplit(' ', 1)[0] for l in f]

    entity_embedding_records = []
    
    for e in tqdm(entity_list, desc="lm_probe entity embeddings"):
        e_toks = tokenizer.tokenize(e)
        for cc, seeds in seed_instances_dict.items():
            cc_phrase = ' '.join(cc.split('_'))
            cc_filled_prompts = [pr.format(cc_phrase, ', '.join(seeds[:-1]), tokenizer.mask_token, seeds[-1]) for pr in probe_prompts]
            for pr in cc_filled_prompts:
                mask_token_id = tokenizer.tokenize(pr).index(tokenizer.mask_token)
                e_span = (mask_token_id, mask_token_id + len(e_toks))
                e_filled_pr = pr.replace(tokenizer.mask_token, e)
                e_pr_toks = tokenizer.tokenize(e_filled_pr)
                assert e_pr_toks[e_span[0]:e_span[1]] == e_toks, f'{e_filled_pr}\n{e_pr_toks}\n{e_toks}'
                
                indexed_tokens = tokenizer.convert_tokens_to_ids(e_pr_toks)
                tokens_tensor = torch.tensor([indexed_tokens])
                segment_idx = tokens_tensor * 0
                tokens_tensor = tokens_tensor.to(device)
                segments_tensors = segment_idx.to(device)

                with torch.no_grad():
                    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
                
                mask = torch.zeros_like(tokens_tensor).to(device)
                mask[0, e_span[0]:e_span[1]] = 1
                pair_embedding = mean_pooling(outputs, mask).detach().cpu().squeeze().numpy().tolist()
                
                entity_embedding_records.append({
                    'concept': cc,
                    'neighbor': e,
                    'embedding': ' '.join([str(x) for x in pair_embedding])
                })
                
    
    print('Saving lm entity embeddings')
    if lm_ent_dest is not None:
        pd.DataFrame(entity_embedding_records).to_csv(lm_ent_dest, index=None)
    
    return entity_embedding_records
    
    
def get_emb_concept_embeddings(emb_src, seed_concepts_path, emb_cc_dest=None, embedding_dim=768,
                               **kwargs):
    seed_concepts_df = load_seed_aligned_concepts(seed_concepts_path)
    seed_concepts_dicts = seed_concepts_df.to_dict('record')
    
    entity_embeddings = load_embeddings(emb_src, embedding_dim)
    entities = entity_embeddings['entity'].tolist()
    embeddings = entity_embeddings['embedding'].tolist()
    entity_emb_dict = dict(zip(entities, embeddings))

    concept_embeddings = {}
    for _cc_dict in tqdm(seed_concepts_dicts, desc="emb concept embeddings"):
        a_concept = _cc_dict['alignedCategoryName']
        seed_instances = _cc_dict['seedInstances']
        
        embs = []
        for inst in seed_instances:
            try:
                embs.append(entity_emb_dict[inst])
            except KeyError:
                print(f"{inst} not found in entity_emb_dict??")
                continue
        if len(embs) == 0:
            continue
        concept_emb = np.mean(embs, axis=0)
        concept_embeddings[a_concept] = concept_emb

    print('Saving emb concept embeddings')
    if emb_cc_dest is not None:
        with open(emb_cc_dest, 'w') as f:
            for _cc, _emb in concept_embeddings.items():
                f.write("{} {}\n".format(_cc, ' '.join([str(x) for x in _emb])))
    
    return concept_embeddings
    
    
if __name__ == "__main__":
    args = parse_arguments()
    if args.emb_src is None:
        args.emb_src = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
    
    if args.embedding_name is not None:
        args.emb_cc_dest = os.path.join(args.dataset_path, f'BERTembed_{args.embedding_name}_concepts.txt')
        args.lm_cc_dest = os.path.join(args.dataset_path, f'BERTembed_{args.embedding_name}_lm_concepts.txt')
        args.lm_ent_dest = os.path.join(args.dataset_path, f'BERTembed_{args.embedding_name}_lm_entities.txt')
    else:
        args.emb_cc_dest = os.path.join(args.dataset_path, f'BERTembed_concepts.txt')
        args.lm_cc_dest = os.path.join(args.dataset_path, f'BERTembed_lm_concepts.txt')
        args.lm_ent_dest = os.path.join(args.dataset_path, f'BERTembed_lm_entities.txt')

    get_emb_concept_embeddings(**vars(args))
    if args.lm_ent_hearst:
        args.lm_ent_dest = args.lm_ent_dest.replace('.txt', '_hearst.csv')
        get_lm_probe_entity_embeddings_hearst(**vars(args))
    else:
        get_lm_probe_entity_embeddings(**vars(args))
    get_lm_probe_concept_embeddings(**vars(args))
    
    
    