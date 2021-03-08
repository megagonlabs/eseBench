#!/bin/bash
DATA=$1
path=$(pwd)
echo $path
LANGUAGE=EN
MIN_SUP=10
THREAD=$2

### Following are the parameters used in auto_phrase.sh
RAW_TRAIN=${RAW_TRAIN:- ../../../data/$DATA/source/sentences.txt}

### Following are the parameters used in phrasal_segmentation.sh
HIGHLIGHT_MULTI=${HIGHLIGHT_MULTI:- 0.5}
HIGHLIGHT_SINGLE=${HIGHLIGHT_SINGLE:- 0.9}
DATA_PATH=${DATA_PATH:- ../../../data/$DATA}
OUTPUT_PATH=${OUTPUT_PATH:- ../../../data/$DATA/intermediate/}

cd ../tools/AutoPhrase

echo ${green}====Create Sentences.txt First===${reset}
python extractCorpus.py $DATA_PATH $THREAD

echo ${green}====Running Phrase_text Generator===${reset}
### Generating Output for Phrasified Dataset ###
python process_segmentation.py --multi $HIGHLIGHT_MULTI --single $HIGHLIGHT_SINGLE --output  $OUTPUT_PATH --mode whole

mv ../../../data/$DATA/intermediate/phrase_dataset_${HIGHLIGHT_MULTI}_${HIGHLIGHT_SINGLE}.txt ../../../data/$DATA/phrase_text.txt
mv ../../../data/$DATA/intermediate/segmentation.txt ../../../data/$DATA/segmentation_corpus.txt

echo ${green}===Running Phrasal Segmentation on Sentences.txt===${reset}
make
echo ${green}==='RAW_TRAIN:' $RAW_TRAIN===${reset}
echo "phrasal_segmentation.sh parameters:" $DATA $RAW_TRAIN $HIGHLIGHT_MULTI $HIGHLIGHT_SINGLE $THREAD
chmod +x ./phrasal_segmentation.sh
./phrasal_segmentation.sh $DATA $RAW_TRAIN $HIGHLIGHT_MULTI $HIGHLIGHT_SINGLE $THREAD

mv models/$DATA/segmentation.txt ../../../data/$DATA/segmentation.txt
