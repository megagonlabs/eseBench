from tqdm import tqdm
import logging
import argparse
import re
import numpy as np

import numpy as np
import os
import torch

from transformers import AutoTokenizer, AutoModel


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--corpus_name', type=str,
                        required=True, help='Name of the corpus')
    parser.add_argument('-et', '--embedding_type', type=str, default='ac', required=True,
                        help='ac if averaged context. Otherwise pt if pooled tokenized')
    parser.add_argument('-m', '--model_path', type=str, required=True, default= 'bert-base-uncased',
                        help='model_path')
    args = parser.parse_args()
    return args


def get_masked_contexts(input_file):
    """Return a (list of) sentence(s) with entity replaced with MASK."""
    pat = '<phrase>((.*?))</phrase>'
    ent_freq = {}
    ent_context = {}
    with open(input_file, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines), desc="loading corpus"):
            line = line.strip()
            entities = [match.group(1) for match in re.finditer(pat, line)]
            for entity in entities:
                context = line.replace('<phrase>' + entity, '</phrase', '[MASK]')
                context = context.replace('<phrase>', '')
                context = context.replace('</phrase>', '')
                print(entity)
                print(context)

                if entity not in ent_freq:
                    ent_freq[entity] = 0
                ent_freq[entity] += 1

                context_lst = ent_context.get(entity, [])
                context_lst.append(entity)
                ent_context[entity] = context_lst
    return ent_freq, ent_context


def get_contexts(input_file):
    """Return a (list of) sentence(s) by entities."""
    pat = '<phrase>((.*?))</phrase>'
    ent_freq = {}
    ent_context = {}
    with open(input_file, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines), desc="loading corpus"):
            line = line.strip()
            entities = [match.group(1) for match in re.finditer(pat, line)]
            for entity in entities:
                context = line.replace('<phrase>', '')
                context = context.replace('</phrase>', '')
                print(entity)
                print(context)

                if entity not in ent_freq:
                    ent_freq[entity] = 0
                ent_freq[entity] += 1

                context_lst = ent_context.get(entity, [])
                context_lst.append(entity)
                ent_context[entity] = context_lst
    return ent_freq, ent_context


def ensure_tensor_on_device(self, **inputs):
    return {name: tensor.to(self.device) for name, tensor in inputs.items()}


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0].cpu() # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def get_vector(hidden_layers, token_index=0, mode='concat', top_n_layers=4):
    if mode == 'concat':
        # concatenate last 4 layer outputs -> returns [batch_size x seq_len x dim]
        # permute(1,0,2) swaps the the batch and seq_len dim , making it easy to return all the vectors for a particular token position
        return torch.cat(hidden_layers[-top_n_layers:], dim=2).permute(1, 0, 2)[token_index]

    if mode == 'average':
        # avg last 4 layer outputs -> returns [batch_size x seq_len x dim]
        return torch.stack(hidden_layers[-top_n_layers:]).mean(0).permute(1, 0, 2)[token_index]

    if mode == 'sum':
        # sum last 4 layer outputs -> returns [batch_size x seq_len x dim]
        return torch.stack(hidden_layers[-top_n_layers:]).sum(0).permute(1, 0, 2)[token_index]

    if mode == 'last':
        # last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers[-1:][0].permute(1, 0, 2)[token_index]

    if mode == 'second_last':
        # last layer output -> returns [batch_size x seq_len x dim]
        return hidden_layers[-2:-1][0].permute(1, 0, 2)[token_index]
    return None


def get_entity_span(context_ids, entity_ids):
    l = len(entity_ids)
    for i in range(len(context_ids)):
        if context_ids[i:i + l] == entity_ids:
            return i, i+l
    return []


def get_avg_context_embeddings(model_path, input_file):
    '''
    mean pooling from sentence-transformers
    :param model_path:
    :param input_file:
    :return:
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mask_token_id = tokenizer.mask_token_id

    ent_freq, ent_context = get_masked_contexts(input_file)
    entity_embeddings = {}
    for entity, en_context_lst in tqdm(ent_context.items(), total=len(ent_context), desc="computing entity-wise embedding"):
        chunks = [en_context_lst[i:i + 100] for i in range(0, len(en_context_lst), 100)]
        all_context_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer.batch_encode_plus(chunk, add_special_tokens=True, max_length=512, return_tensors='pt', padding=True, truncation=True)
            mask = encoded_input['input_ids'] != mask_token_id
            with torch.no_grad():
                encoded_input = ensure_tensor_on_device(**encoded_input)
                model_output = model(**encoded_input)  # Compute token embeddings
            context_embeddings = mean_pooling(model_output, mask)  # mean pooling
            all_context_embeddings.append(context_embeddings)
        entity_embedding = torch.mean(torch.cat(all_context_embeddings, dim=0), dim=0).cpu().detach().numpy().tolist()
        entity_embeddings[entity] = entity_embedding
    return entity_embeddings, ent_freq


def get_pooled_token_embeddings(model_path, input_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    ent_freq, ent_context = get_masked_contexts(input_file)
    entity_embeddings = {}
    for entity, en_context_lst in tqdm(ent_context.items(), total=len(ent_context), desc="computing entity-wise embedding"):
        entity_ids = tokenizer.encode(entity)
        chunks = [en_context_lst[i:i + 100] for i in range(0, len(en_context_lst), 100)]
        all_context_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer.batch_encode_plus(chunk, add_special_tokens=True, max_length=512, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                model_output = model(**encoded_input)  # Holds the list of 12 layer embeddings for each token
                hidden_states = model_output[2]  # [batch_size x seq_length x vector_dim]

                for i, context in enumerate(chunk):
                    input_ids = encoded_input[i]['input_ids']
                    entity_start_token_index, entity_end_token_index = get_entity_span(input_ids, entity_ids)

                    entity_vecs = []
                    for index in range(entity_start_token_index, entity_end_token_index):
                        vec = get_vector(hidden_states[i], index, mode='concat', top_n_layers=4)
                        entity_vecs.append(vec)
                    context_embedding = torch.mean(entity_vecs, dim=0).cpu().detach().numpy().tolist()
                    all_context_embeddings.append(context_embedding)
        entity_embedding = torch.mean(torch.cat(all_context_embeddings, dim=0), dim=0).cpu().detach().numpy().tolist()
        entity_embeddings[entity] = entity_embedding
    return entity_embeddings, ent_freq


def main():
    args = parse_arguments()
    args.input_file = os.path.join(args.corpus_name, 'sent_segmentation.txt')
    args.embed_dest = os.path.join(args.corpus_name, 'BERTembed.txt')
    args.embed_num = os.path.join(args.corpus_name, 'BERTembednum.txt')

    if args.embedding_type == 'ac':
        entity_embeddings, ent_freq = get_avg_context_embeddings(args.model_path, args.input_file)
    elif args.embedding_type == 'pt':
        entity_embeddings, ent_freq = get_pooled_token_embeddings(args.model_path, args.input_file)

    print("Saving embedding")
    with open(args.embed_dest, 'w') as f, open(args.embed_num, 'w') as f2:
        for eid in entity_embeddings:
            f.write("{} {}\n".format(eid, ' '.join([str(x) for x in entity_embeddings[eid]])))
            f2.write("{} {}\n".format(eid, ent_freq[eid]))


if __name__ == "__main__":
    main()


