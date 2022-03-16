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

from .compute_concept_clusters import load_embeddings

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default=None,
                        help='Dataset path with intermediate output')
    parser.add_argument('-b', '--benchmark_dir', type=str, default=None,
                        help='Benchmark directory path')
    parser.add_argument('-e', '--embed_src', type=str, default=None,
                        help='Dataset path with pre-computed embeddings')
    parser.add_argument('-sc', '--seed_aligned_concept_src', type=str, default=None,
                        help='Seed (aligned) concepts file path')
    parser.add_argument('-c', '--thread_ct', type=int, default=1,
                        help='No. of threads')
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
    parser.add_argument('-kdt', '--kdt', action='store_true',
                        help='whether using kd-tree (Annoy)')

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

def get_concept_knn(embed_src, embedding_dim, seed_aligned_concept_src, cluster_size, thread_ct, cluster_dest, kdt, **kwargs):
    seed_concepts_df = load_seed_aligned_concepts(seed_aligned_concept_src)
    seed_concepts_dicts = seed_concepts_df.to_dict('record')
    
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    entities = entity_embeddings['entity'].tolist()
    embeddings = entity_embeddings['embedding'].tolist()
    entity_emb_dict = dict(zip(entities, embeddings))

    if cluster_size is None:
        cluster_size = len(entities)
    
    neighbors = []
    
    if kdt:
        t = AnnoyIndex(embedding_dim, 'angular')
        for i, row in tqdm(entity_embeddings.iterrows(), total=entity_embeddings.shape[0], desc="building entity index"):
            t.add_item(i, row['embedding'])
        t.build(100)
    else:
        t = None
        
#     for i, (a_concept, u_concept, gnrl, seed_instances) in tqdm(seed_concepts_df.iterrows(), desc="finding nearest neighbors by concept"):
    for i, _cc_dict in tqdm(list(enumerate(seed_concepts_dicts))):
        a_concept = _cc_dict['alignedCategoryName']
        seed_instances = _cc_dict['seedInstances']
        seed_instances = [e.lower() for e in seed_instances]
        
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

        if kdt:
            nns, dists = t.get_nns_by_vector(concept_emb, cluster_size, include_distances=True)
            cos_sim_scores = [(2 - d ** 2) / 2 for d in dists]  # convert angular distance to cosine similarity
        else:
            all_cos_sim_scores = cosine_similarity([concept_emb.tolist()], embeddings)[0]
#             _partition = np.argpartition(-all_cos_sim_scores, kth=cluster_size)
#             nns = _partition[:cluster_size]
            _ranking = np.argsort(-all_cos_sim_scores)
            nns = _ranking[:cluster_size]
            cos_sim_scores = [all_cos_sim_scores[i] for i in nns]
            
        zipped = list(zip(nns, cos_sim_scores))
        sorted_nns = sorted(zipped, key=lambda x: x[1], reverse=True)
        if len(sorted_nns) > 0:
            for nn_idx, d in sorted_nns:
                neighbor_entity = entities[nn_idx]
                if neighbor_entity in seed_instances:
                    continue
                neighbors.append({"concept": a_concept, "neighbor": neighbor_entity, "sim": d})
   
    c_df = pd.DataFrame(neighbors)
    c_df.to_csv(cluster_dest, index=None)
    
    
def main():
    args = parse_arguments()
#     args.input_file = os.path.join(args.dataset_dir, 'sent_segmentation.txt')
#     args.embed_num = os.path.join(args.dataset_dir, 'BERTembednum+seeds.txt')
#     args.seed_aligned_concept_src = os.path.join(args.benchmark_dir, 'seed_aligned_concepts.csv')

    if args.embed_src is None:
        args.embed_src = os.path.join(args.dataset_dir, 'BERTembed+seeds.txt')

    if args.seed_aligned_concept_src is None:
        args.seed_aligned_concept_src = os.path.join(args.benchmark_dir, 'seed_aligned_concepts.csv')
#     if args.aux_concepts:
#         args.seed_aligned_concept_src = os.path.join(args.benchmark_dir, 'seed_aligned_concepts_aux.csv')
#     else:
#         args.seed_aligned_concept_src = os.path.join(args.benchmark_dir, 'seed_aligned_concepts.csv')
    
    get_concept_knn(**vars(args))


if __name__ == "__main__":
    main()