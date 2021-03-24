#!/bin/bash
chmod +x lm_concept_learn.sh
algo=kmeans

for data in yelp indeeda
do
    for embed in meg-ac meg-pt corel
    do
        for size in 800 1000 
        do
            if [ "$embed" = "meg-pt" ]; then
                ./lm_concept_learn.sh $algo $data $embed $size 3072
            else
                ./lm_concept_learn.sh $algo $data $embed $size 768
            fi
        done
    done
done
