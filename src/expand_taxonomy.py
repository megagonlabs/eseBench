import argparse
from collections import defaultdict
import pandas as pd
import os

from concept_learning import compute_concept_seeds_knn
from concept_learning import compute_EE_LM_probe

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, default='indeed_job_descriptions',
                        help='Dataset path with intermediate output')
    parser.add_argument('-c', '--thread_ct', type=int, default=1,
                        help='No. of threads')
    parser.add_argument('-dim', '--embedding_dim', type=int, default=768,
                        help='embedding_dim')
    parser.add_argument('-kdt', '--kdt', action='store_true',
                        help='whether using kd-tree (Annoy)')
    parser.add_argument('--template_agg', type=str)
    parser.add_argument('--tmpl_agg_func', type=str)


    args = parser.parse_args()
    return args

def mrr(ee_LM_df, ee_emb_df, output_dir):
    concept_list = ee_LM_df['concept'].drop_duplicates().tolist()
    ee_mrr_combine_list = []

    for _cc in sorted(concept_list):
        _ce_df = ee_emb_df[ee_emb_df['concept'] == _cc].sort_values(by='sim', ascending=False)
        _ee_emb_list = _ce_df['neighbor'].tolist()
        _ee_LM_list = ee_LM_df[ee_LM_df['concept'] == _cc]['neighbor'].tolist()
            
        _all_entities_mrr = defaultdict(float)
        for i, _e in enumerate(_ee_emb_list):
            _all_entities_mrr[_e] += 1.0 / (i+1)
        for i, _e in enumerate(_ee_LM_list):
            _all_entities_mrr[_e] += 1.0 / (i+1)

        _all_entities_mrr_list = sorted(list(_all_entities_mrr.items()), key=lambda p: p[-1], reverse=True)
        
        for _e, _mrr in _all_entities_mrr_list:
            ee_mrr_combine_list.append((_cc, _e, _mrr))
    
    df = pd.DataFrame(ee_mrr_combine_list, columns=['concept', 'neighbor', 'MRR'])
    df = df.merge(ee_LM_df, how='left', on=['concept', 'neighbor'])
    df = df.merge(ee_emb_df, how='left', on=['concept', 'neighbor'])
    df.to_csv(os.path.join(output_dir, 'ee_mrr_combine_bert_k=None.csv'), index=None)
    df = pd.read_csv(os.path.join(output_dir, 'ee_mrr_combine_bert_k=None.csv'))
    mrr = df.groupby('concept').head(200)
    mrr.to_csv(os.path.join(output_dir, 'ee_mrr_combine_bert_k=200.csv'), index=None)

def main():
    args = parse_arguments()

    embed_src = os.path.join('../data', args.dataset_name, 'intermediate/BERTembed+seeds.txt')
    if not os.path.exists(embed_src):
        embed_src = os.path.join('../data', args.dataset_name, 'intermediate/BERTembed.txt')
    seed_aligned_concept_src = os.path.join('../data', args.dataset_name, 'seed_aligned_concepts.csv')  
    benchmark_dir = os.path.join('../data', args.dataset_name)

    concept_knn_path = os.path.join('../data', args.dataset_name, 'intermediate/ee_concept_knn_k=None.csv')
    compute_concept_seeds_knn.get_concept_knn(embed_src=embed_src, benchmark_dir=benchmark_dir, seed_aligned_concept_src=seed_aligned_concept_src, cluster_size=None, cluster_dest=concept_knn_path, **vars(args))

    args.template_agg == 'max'
    args.tmpl_agg_func = lambda l : max(l)
    concepts_lm_path = os.path.join('../data', args.dataset_name, 'intermediate/ee_LM_bert_k=None.csv')
    compute_EE_LM_probe.EE_LMProbe(seed_concepts_path=seed_aligned_concept_src, benchmark_dir=benchmark_dir, emb_path=embed_src, lm_probe_type='bert', lm_probe_model='bert-base-uncased',topk=None, dest=concepts_lm_path, **vars(args))

    ee_LM_path = os.path.join('../data', args.dataset_name, 'intermediate/ee_LM_bert_k=None.csv')

    ee_LM_df = pd.read_csv(ee_LM_path)
    ee_emb_df = pd.read_csv(concept_knn_path)

    mrr(ee_LM_df, ee_emb_df,  os.path.join('../data', args.dataset_name, 'intermediate'))

if __name__ == "__main__":
    main()
