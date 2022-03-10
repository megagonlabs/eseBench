#!/bin/bash
DATA=$1

source activate /nfs/users/nikita/.conda/envs/kb_entity_expan

cd keyword_extraction
chmod +x ./corpusProcess.sh
./corpusProcess.sh $DATA 8

cd ../concept_learning

module load slurm

cmd1="/nfs/users/nikita/.conda/envs/kb_entity_expan/bin/python compute_keyphrase_embeddings.py -m bert-base-uncased -et ac -d ../../data/${DATA}/intermediate"
cmd2="/nfs/users/nikita/.conda/envs/kb_entity_expan/bin/python add_seed_instances_embeddings.py -m bert-base-uncased -et ac -d ../../data/${DATA}/intermediate -b ../../data/${DATA}/ -c 750"
cmd3="/nfs/users/nikita/.conda/envs/kb_entity_expan/bin/python expand_taxonomy.py -d ${DATA} --kdt"

sbatch -G 1 -J kb_expan --ntasks-per-node=1 --nodes=1 -w node002 --wrap="$cmd1 && $cmd2 && $cmd3"