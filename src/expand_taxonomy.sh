#!/bin/bash
ENV_PATH=$1
DATA=$2

source activate $ENV_PATH

${ENV_PATH}/bin/python compute_keyphrase_embeddings.py -m bert-base-uncased -et ac -d ../../data/${DATA}/intermediate
${ENV_PATH}/bin/python add_seed_instances_embeddings.py -m bert-base-uncased -et ac -d ../../data/${DATA}/intermediate -b ../../data/${DATA}/ -c 750
${ENV_PATH}/bin/python expand_taxonomy.py -d ${DATA} --kdt