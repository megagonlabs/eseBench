set -e

wiki_data_dir=/mnt/efs/shared/meg_shared_scripts/meg-kb/data/wiki-meg-ac/intermediate
indeed_data_dir=/mnt/efs/shared/meg_shared_scripts/meg-kb/data/indeeda-meg-ac/intermediate

# CUDA_VISIBLE_DEVICES=2 python multiview_EE_train.py -d $indeed_data_dir -dn indeed -o_dir ${indeed_data_dir}/cotraining_results -ep 500 -g_clip 0.01 -w_emb 1 -w_lm 1 -w_joint 1 -w_diff 1 -rsmp -hearst -loss rank -r_margin 0.5 -v 17.1

# CUDA_VISIBLE_DEVICES=2 python multiview_EE_train.py -d $wiki_data_dir -o_dir ${wiki_data_dir}/cotraining_results -ep 500 -g_clip 0.01 -w_emb 1 -w_lm 1 -w_joint 1 -w_diff 1 -rsmp -hearst -loss rank -r_margin 0.5 -v 17.1

CUDA_VISIBLE_DEVICES=2 python multiview_EE_train.py -d $indeed_data_dir -dn indeed -o_dir ${indeed_data_dir}/cotraining_results -ep 500 --lm_ent_hearst -loss rank -r_margin 0.5 -v 17.1-indeed --test_only -ckpt=lightning_logs/default/17.1-indeed/checkpoints/epoch=499-step=123499.ckpt

# CUDA_VISIBLE_DEVICES=2 python multiview_EE_train.py -d $wiki_data_dir -o $wiki_data_dir/ee_clf_test_v=15.4.csv -ep 500 -g_clip 0.01 -w_emb 1 -w_lm 0 -w_joint 0 -w_diff 0 --lm_ent_hearst -loss rank -r_margin 0.5 -v 15.4
