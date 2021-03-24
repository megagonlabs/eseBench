#!/bin/bash
CLUS_ALGO=$1
DATA=$2
EMBED=$3
CLUS_SIZE=$4
DIM=$5
DATA_DIR=${DATA}-${EMBED}

echo ${green}==='Corpus Name:' $DATA_DIR===${reset}
echo ${green}==='Algo:' $CLUS_ALGO', Num clusters:' $CLUS_SIZE', dim:' $DIM ===${reset}
INPUT=${INPUT:- ../../data/$DATA_DIR/intermediate/}
OUTFILE=${CLUS_ALGO}_${CLUS_SIZE}.csv
OUTPUT=${OUTPUT:- ../../data/$DATA_DIR/intermediate/$OUTFILE}

echo ${green}==='Input path:' $INPUT===${reset}
echo ${green}==='Output path:' $OUTPUT===${reset}

python compute_concept_clusters.py -d $INPUT -ca $CLUS_ALGO -s $CLUS_SIZE -dim $DIM -o $OUTPUT
