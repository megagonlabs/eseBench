#!/bin/bash
for data in tripadvisor 
do
    for embed in ac pt
    do
        DATA=${data}-meg-${embed}
        DATA_DIR=${DATA_DIR:- ../../data/$DATA/intermediate}
        python compute_keyphrase_embeddings.py -m bert-base-uncased -et $embed -d $DATA_DIR -c 750
    done
done
