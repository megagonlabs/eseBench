from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
import torch
import math
from annoy import AnnoyIndex

from compute_concept_clusters import load_embeddings

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_path', type=str,
                        required=True, help='Benchmark directory path')
#     parser.add_argument('-c', '--thread_ct', type=int, default=1,
#                         help='No. of threads')
#     parser.add_argument('-ca', '--clustering_algorithm', type=str, default='kmeans',
#                         help='agg or kmeans or knn')
    parser.add_argument('-aux_cc', '--aux_concepts', action='store_true',
                        help='including auxiliary concepts')
    parser.add_argument('-s', '--cluster_size', type=int, default=None,
                        help='cluster size for kmeans or min cluster size for hdbscan')
    parser.add_argument('-o', '--cluster_dest', type=str, required=True,
                        help='Path to clusters')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')
#     parser.add_argument('-kdt', '--kdt', action='store_true',
#                         help='whether using kd-tree (Annoy)')
    
    args = parser.parse_args()
    return args

def load_seed_aligned_concepts(path):
    df = pd.read_csv(path)
    df = df[df["generalizations"] != "x"]
    df["seedInstances"] = df["seedInstances"].map(lambda s : eval(str(s)))
    return df

def load_seed_aligned_relations(path):
    df = pd.read_csv(path)
    df = df[df["range"] != "x"]
    return df


def get_concept_contrastive_knn(embed_src, embedding_dim, seed_aligned_concept_src, cluster_size, cluster_dest, **kwargs):
    
    seed_concepts_df = load_seed_aligned_concepts(seed_aligned_concept_src)
    seed_concepts_dicts = seed_concepts_df.to_dict('record')
    
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    entities = entity_embeddings['entity'].tolist()
    embeddings = entity_embeddings['embedding'].tolist()
    entity_emb_dict = dict(zip(entities, embeddings))

    neighbors = []
    
    concept_emb_dict = dict()
#     for i, (a_concept, u_concept, gnrl, seed_instances) in tqdm(seed_concepts_df.iterrows(), desc="finding nearest neighbors by concept"):
    for i, _cc_dict in tqdm(list(enumerate(seed_concepts_dicts))):
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
        concept_emb_dict[a_concept] = concept_emb
    
    concepts = list(concept_emb_dict.keys())
    concept_embs = list(concept_emb_dict.values())

    # (n_concepts, n_entities)
    cos_matrix = cosine_similarity(concept_embs, embeddings)
    
    cands_for_concepts = [[] for _ in range(len(concepts))]
    for e_id, e in enumerate(entities):
        _scores = cos_matrix[:, e_id]
        _cc_ranking = np.argsort(-_scores)
        _max_cc_id = _cc_ranking[0]
        _second_cc_id = _cc_ranking[1]
        _score = _scores[_max_cc_id]
        _2nd_score = _scores[_second_cc_id]
        _margin = _score - _2nd_score
        cands_for_concepts[_max_cc_id].append({
            'concept': concepts[_max_cc_id],
            '2nd_concept': concepts[_second_cc_id],
            'neighbor': e,
            'sim': _score,
            '2nd_sim': _2nd_score,
            'margin': _margin,
            'sim+margin': _score + _margin
        })
    
#         _partition = np.argpartition(-all_cos_sim_scores, kth=cluster_size)
#         nns = _partition[:cluster_size]
#         cos_sim_scores = [all_cos_sim_scores[i] for i in nns]
            
#         zipped = list(zip(nns, cos_sim_scores))
#         sorted_nns = sorted(zipped, key=lambda x: x[1], reverse=True)
#         if len(sorted_nns) > 0:
#             for nn_idx, d in sorted_nns:
#                 neighbor_entity = entities[nn_idx]
#                 if neighbor_entity in seed_instances:
#                     continue
#                 neighbors.append({"concept": a_concept, "neighbor": neighbor_entity, "sim": d})
   
    for cc_id, cands in enumerate(cands_for_concepts):
        cands_sorted = sorted(cands, key=lambda d: d['sim+margin'], reverse=True)
        neighbors.extend(cands_sorted[:cluster_size])
    
    c_df = pd.DataFrame(neighbors)
    c_df.to_csv(cluster_dest, index=None)
    
    
def main():
    print("Warning: no longer maintained. Use post_proc contrastive")
    args = parse_arguments()
#     args.input_file = os.path.join(args.dataset_path, 'sent_segmentation.txt')
    args.embed_src = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
#     args.embed_num = os.path.join(args.dataset_path, 'BERTembednum+seeds.txt')
    if args.aux_concepts:
        args.seed_aligned_concept_src = os.path.join(args.benchmark_path, 'seed_aligned_concepts_aux.csv')
    else:
        args.seed_aligned_concept_src = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')
    
    get_concept_contrastive_knn(**vars(args))


if __name__ == "__main__":
    main()