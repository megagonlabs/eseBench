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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_path', type=str,
                        required=True, help='Dataset path with intermediate output')
    parser.add_argument('-c', '--thread_ct', type=int, default=4,
                        help='No. of threads')
    parser.add_argument('-ca', '--clustering_algorithm', type=str, default='kmeans',
                        help='agg or kmeans or knn')
    parser.add_argument('-s', '--cluster_size', type=int, default=500,
                        help='cluster size for kmeans or min cluster size for hdbscan')
    parser.add_argument('-o', '--cluster_dest', type=str, required=True,
                        help='Path to clusters')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')

    args = parser.parse_args()
    return args


def normalize(x):
    return x / x.norm(dim=1)[:, None]


def cosine(x, y=None, norm=False):
    """
    x: LHS (ie, your query documents)
    y: RHS to compare against (ie, the database). If not present, you're performing pairwise_cosine (x cosine x)
    norm: normalize output? something to fiddle with. I've found it improves Agglomorative Clustering for some reason?
    """
    x = torch.tensor(x)
    if y is None:
        x = y = normalize(x)
    else:
        y = torch.tensor(y)
        # normalize together first
        both = torch.cat((x, y), 0)
        both = normalize(both)
        x, y = both[:x.shape[0]], both[x.shape[0]:]
    sim = torch.mm(x, y.T)
    if norm: sim = normalize(sim)
    # this part key. For sklearn hierarchical clustering models, value must be greater than 1. I've fiddled with all sorts
    # of positive-ify cosine similarities (which is naturally between [-1 1]). This includes absolute(), sim.acos()/math.pi, etc.
    sim = (sim + 1.) / 2.
    # Prefer working with dist than sim, since most default sorting (eg numpy, pandas) is ascending
    dist = 1. - sim
    return dist.numpy()


def agg(embed_src, embedding_dim, cluster_size, thread_ct, cluster_dest, **kwargs):
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    embeddings = entity_embeddings['embedding'].tolist()
    dists = cosine(embeddings, norm=True)
    # nc = math.floor(1 + 4 * math.log10(dists.shape[0]))
    agg = AgglomerativeClustering(n_clusters=cluster_size, affinity='precomputed', linkage='average')
    clus_id = agg.fit_predict(dists)
    c_df = pd.DataFrame({"entity": entity_embeddings['entity'], "clus_id": clus_id})
    c_df = c_df.sort_values(by='clus_id')
    c_df.to_csv(cluster_dest, index=None)


def knn(embed_src, embedding_dim, cluster_size, thread_ct, cluster_dest, **kwargs):
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    t = AnnoyIndex(768, 'angular')
    entities = entity_embeddings['entity'].tolist()
    for i, row in tqdm(entity_embeddings.iterrows(), total=entity_embeddings.shape[0], desc="building entity index"):
        t.add_item(i, row['embedding'])
    t.build(100)

    neighbors = []
    for i, entity in enumerate(tqdm(entities, desc="finding nearest neighbors by entity")):
        nns, dists = t.get_nns_by_item(i, cluster_size + 1, include_distances=True)
        cos_sim_scores = [(2 - d ** 2) / 2 for d in dists]  # convert angular distance to cosine similarity
        zipped = list(zip(nns, cos_sim_scores))
        sorted_nns = sorted(zipped, key=lambda x: x[1], reverse=True)
        if len(sorted_nns) > 0:
            for nn_idx, d in sorted_nns:
                neighbor_entity = entities[nn_idx]
                if neighbor_entity == entity:
                    continue
                neighbors.append({"entity": entity, "neighbor": neighbor_entity, "sim": d})
    c_df = pd.DataFrame(neighbors)
    c_df.to_csv(cluster_dest, index=None)


def kmeans(embed_src, embedding_dim, cluster_size, thread_ct, cluster_dest, **kwargs):
    entity_embeddings = load_embeddings(embed_src, embedding_dim)
    print("Clustering...")
    X = np.array(entity_embeddings['embedding'].tolist())
    clf = KMeans(n_clusters=cluster_size, n_jobs=thread_ct)
    clus_id = clf.fit_predict(X)
    print("Done.")
    c_df = pd.DataFrame({"entity": entity_embeddings['entity'], "clus_id": clus_id})
    c_df = c_df.sort_values(by='clus_id')
    c_df.to_csv(cluster_dest, index=None)


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


def main():
    args = parse_arguments()
    args.input_file = os.path.join(args.dataset_path, 'sent_segmentation.txt')
    args.embed_src = os.path.join(args.dataset_path, 'BERTembed.txt')
    args.embed_num = os.path.join(args.dataset_path, 'BERTembednum.txt')

    if args.clustering_algorithm == 'kmeans':
        kmeans(**vars(args))
    elif args.clustering_algorithm == 'agg':
        agg(**vars(args))
    elif args.clustering_algorithm == 'knn':
        knn(**vars(args))


if __name__ == "__main__":
    main()


