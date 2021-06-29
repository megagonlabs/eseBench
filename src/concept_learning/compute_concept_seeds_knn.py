from tqdm import tqdm
import logging
import argparse
from sklearn.cluster import KMeans, AgglomerativeClustering
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
    parser.add_argument('-c', '--thread_ct', type=int, default=1,
                        help='No. of threads')
#     parser.add_argument('-ca', '--clustering_algorithm', type=str, default='kmeans',
#                         help='agg or kmeans or knn')
    parser.add_argument('-s', '--cluster_size', type=int, default=1000,
                        help='cluster size for kmeans or min cluster size for hdbscan')
    parser.add_argument('-o', '--cluster_dest', type=str, required=True,
                        help='Path to clusters')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')

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

def get_concept_knn(embed_src, embedding_dim, seed_aligned_concept_src, cluster_size, thread_ct, cluster_dest, **kwargs):
    seed_concepts_df = load_seed_aligned_concepts(seed_aligned_concept_src)
    
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    t = AnnoyIndex(embedding_dim, 'angular')
    entities = entity_embeddings['entity'].tolist()
    for i, row in tqdm(entity_embeddings.iterrows(), total=entity_embeddings.shape[0], desc="building entity index"):
        t.add_item(i, row['embedding'])
    t.build(100)
    
    entity_emb_dict = dict(zip(entities, entity_embeddings['embedding'].tolist()))

    neighbors = []
    for i, (a_concept, u_concept, gnrl, seed_instances) in tqdm(seed_concepts_df.iterrows(), desc="finding nearest neighbors by concept"):
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
        
        nns, dists = t.get_nns_by_vector(concept_emb, cluster_size + 1, include_distances=True)
        cos_sim_scores = [(2 - d ** 2) / 2 for d in dists]  # convert angular distance to cosine similarity
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
    args.input_file = os.path.join(args.dataset_path, 'sent_segmentation.txt')
    args.embed_src = os.path.join(args.dataset_path, 'BERTembed+seeds.txt')
#     args.embed_num = os.path.join(args.dataset_path, 'BERTembednum+seeds.txt')
    args.seed_aligned_concept_src = os.path.join(args.benchmark_path, 'seed_aligned_concepts.csv')

    get_concept_knn(**vars(args))


if __name__ == "__main__":
    main()