#!/bin/bash
chmod +x lm_concept_learn.sh
algo=knn

for data in imdb 
do
    for embed in meg-ac
    do
        for size in 20
        do
            #if [ "$embed" = "meg-pt" ]; then
            #    ./lm_concept_learn.sh $algo $data $embed $size 3072 1
            #else
            ./lm_concept_learn.sh $algo $data $embed $size 768 1
            #fi
        done
    done
done
