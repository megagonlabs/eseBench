#!/bin/bash
DATA=$1
GPU=$2

cd keyword_extraction
chmod +x ./corpusProcess.sh
./corpusProcess.sh $DATA 8

cd ../concept_learning
CUDA_VISIBLE_DEVICES=$GPU python compute_keyphrase_embeddings.py -m bert-base-uncased -et ac -d ../../data/$DATA/intermediate -c 750



