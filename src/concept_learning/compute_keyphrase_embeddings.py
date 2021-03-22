from tqdm import tqdm
import argparse
import re
import numpy as np
import random
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-et', '--embedding_type', type=str, default='ac', required=True,
                        help='ac if averaged context. Otherwise pt if pooled tokenized')
    parser.add_argument('-m', '--model_path', type=str, required=True, default= 'bert-base-uncased',
                        help='model_path')
    parser.add_argument('-c', '--max_context_ct', type=int, default=500, help='max. no. of context to consider')
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
            line = str(line.strip())
            entities = [match.group(1) for match in re.finditer(pat, line)]
            for entity in entities:
                context = line.replace('<phrase>' + entity + '</phrase>', '[MASK]')
                context = context.replace('<phrase>', '')
                context = context.replace('</phrase>', '')
                c = context.split('[MASK]')
                if len(c) != 2:  # sanity to not have too many repeating phrases in the context
                    continue

                # ignore too short contexts
                if len(context) < 15:
                    continue

                # print(entity)
                # print(context)

                if entity not in ent_freq:
                    ent_freq[entity] = 0
                ent_freq[entity] += 1

                context_lst = ent_context.get(entity, [])
                context_lst.append(context)
                ent_context[entity] = context_lst
    dedup_context = {}
    for e, v in ent_context.items():
        dedup_context[e] = list(set(v))
        # print(e)
        # print(len(list(set(v))))
    return ent_freq, dedup_context


def get_contexts(input_file):
    """Return a (list of) sentence(s) by entities."""
    pat = '<phrase>((.*?))</phrase>'
    ent_freq = {}
    ent_context = {}
    with open(input_file, "r") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines), desc="loading corpus"):
            line = str(line.strip())
            entities = [match.group(1) for match in re.finditer(pat, line)]
            for entity in entities:
                context = line.replace('<phrase>', '')
                context = context.replace('</phrase>', '')

                # ignore too short contexts
                if len(context) < 15:
                    continue

                # print(entity)
                # print(context)

                if entity not in ent_freq:
                    ent_freq[entity] = 0
                ent_freq[entity] += 1

                context_lst = ent_context.get(entity, [])
                context_lst.append(context)
                ent_context[entity] = context_lst
    dedup_context = {}
    for e, v in ent_context.items():
        dedup_context[e] = list(set(v))
    return ent_freq, dedup_context


def ensure_tensor_on_device(device, **inputs):
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    token_embeddings = model_output[0].cpu() # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
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


def get_avg_context_embeddings(model_path, input_file, max_context_ct):
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
        en_context_lst = random.sample(en_context_lst, min(len(en_context_lst), max_context_ct))
        chunks = [en_context_lst[i:i + 100] for i in range(0, len(en_context_lst), 100)]
        # print(entity)
        # print(len(en_context_lst))
        all_context_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer.batch_encode_plus(chunk, return_token_type_ids=True, add_special_tokens=True, max_length=128, return_tensors='pt', padding=True, pad_to_max_length=True, truncation=True)
            mask = encoded_input['input_ids'] != mask_token_id
            with torch.no_grad():
                encoded_input = ensure_tensor_on_device(device, **encoded_input)
                model_output = model(**encoded_input)  # Compute token embeddings
            context_embeddings = mean_pooling(model_output, mask)  # mean pooling
            all_context_embeddings.append(context_embeddings)
        entity_embedding = torch.mean(torch.cat(all_context_embeddings, dim=0), dim=0).cpu().detach().numpy().tolist()
        entity_embeddings[entity] = entity_embedding
    return entity_embeddings, ent_freq


def get_pooled_token_embeddings(model_path, input_file, max_context_ct):
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mask_token_id = tokenizer.mask_token_id

    ent_freq, ent_context = get_masked_contexts(input_file)
    entity_embeddings = {}
    for entity, en_context_lst in tqdm(ent_context.items(), total=len(ent_context), desc="computing entity-wise embedding"):
        en_context_lst = random.sample(en_context_lst, min(len(en_context_lst), max_context_ct))
        chunks = [en_context_lst[i:i + 200] for i in range(0, len(en_context_lst), 200)]
        all_context_embeddings = []
        for chunk in chunks:
            encoded_input = tokenizer.batch_encode_plus(chunk, return_token_type_ids=True, add_special_tokens=True, max_length=128, return_tensors='pt', padding=True, pad_to_max_length=True, truncation=True)
            with torch.no_grad():
                encoded_input_tensor = ensure_tensor_on_device(device, **encoded_input)
                model_output = model(**encoded_input_tensor)  # Holds the list of 12 layer embeddings for each token
                hidden_states = model_output[2]  # [batch_size x seq_length x vector_dim]
                all_hidden_embeddings = torch.stack(hidden_states, dim=0) # tuple of 13 layers to tensor [layer x batch_size x seq_length x vector_dim]
                # print(all_hidden_embeddings.size())
                all_hidden_embeddings = all_hidden_embeddings.permute(1, 0, 2, 3) # switch layer and batch [batch_size x layer x seq_length x vector_dim]
                # print(all_hidden_embeddings.size())
                for i, context in enumerate(chunk):
                    try:
                        ith_input_ids = encoded_input['input_ids'][i].cpu().detach().numpy().tolist()
                        if mask_token_id in ith_input_ids:
                            mask_id = ith_input_ids.index(mask_token_id)
                            # print('mask id {}'.format(mask_id))
                            ith_hidden_states = all_hidden_embeddings[i]
                            # print('hidden states size {}'.format(ith_hidden_states.size()))
                            context_embedding = get_vector(ith_hidden_states, mask_id, mode='concat',
                                                           top_n_layers=4)
                            if len(context_embedding) == 3072:  # 768 * 4
                                all_context_embeddings.append(context_embedding)
                            else:
                                print(len(context_embedding))
                        else:
                            # print('####NOT FOUND#####')
                    except IndexError:
                        pass
                    except KeyError:
                        pass
                    except ValueError:
                        pass
                    # context_embedding = torch.mean(entity_vecs, dim=0).cpu().detach().numpy().tolist()
                    # all_context_embeddings.append(context_embedding)
        if len(all_context_embeddings) > 0:
            entity_embedding = torch.mean(torch.stack(all_context_embeddings), dim=0).cpu().detach().numpy().tolist()
            entity_embeddings[entity] = entity_embedding
    return entity_embeddings, ent_freq


def main():
    args = parse_arguments()
    args.input_file = os.path.join(args.dataset_path, 'sent_segmentation.txt')
    args.embed_dest = os.path.join(args.dataset_path, 'BERTembed.txt')
    args.embed_num = os.path.join(args.dataset_path, 'BERTembednum.txt')

    if args.embedding_type == 'ac':
        entity_embeddings, ent_freq = get_avg_context_embeddings(args.model_path, args.input_file, args.max_context_ct)
    elif args.embedding_type == 'pt':
        entity_embeddings, ent_freq = get_pooled_token_embeddings(args.model_path, args.input_file, args.max_context_ct)

    print("Saving embedding")
    with open(args.embed_dest, 'w') as f, open(args.embed_num, 'w') as f2:
        for eid in entity_embeddings.keys():
            f.write("{} {}\n".format(eid, ' '.join([str(x) for x in entity_embeddings[eid]])))
            f2.write("{} {}\n".format(eid, ent_freq[eid]))


if __name__ == "__main__":
    main()


