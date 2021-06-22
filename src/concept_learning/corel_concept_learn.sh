#!/bin/bash
DATA=$1
path=$(pwd)
echo $path
LANGUAGE=EN
TOPIC=$2

green=`tput setaf 2`
reset=`tput sgr0`
echo ${green}==='Corpus Name:' $DATA===${reset}
echo ${green}==='Current Path:' $path===${reset}

cd ../tools/corel/c
echo ${green}===Generate Concept Clusters===${reset}
chmod +x run_emb_part_tax.sh
./run_emb_part_tax.sh $DATA $TOPIC
cd ../..
