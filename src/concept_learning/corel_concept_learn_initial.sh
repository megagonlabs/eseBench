#!/bin/bash
DATA=$1
path=$(pwd)
echo $path
LANGUAGE=EN
MIN_SUP=10
THREAD=$2
GPU=$3
TOPIC=$4

green=`tput setaf 2`
reset=`tput sgr0`
echo ${green}==='Corpus Name:' $DATA===${reset}
echo ${green}==='Current Path:' $path===${reset}

EMBED_PATH=${EMBED_PATH:- ../../../data/$DATA/intermediate/}

echo ${green}===Generate BERT embeddings===${reset}
cd ../tools/AutoPhrase
CUDA_VISIBLE_DEVICES=$GPU python extractBertEmbedding.py $EMBED_PATH $THREAD

echo ${green}===Generate Concept Clusters===${reset}
cd ../corel/c
chmod +x run_emb_part_tax.sh
./run_emb_part_tax.sh $DATA $TOPIC
cd ../..