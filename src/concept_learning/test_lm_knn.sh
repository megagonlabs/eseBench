#!/bin/bash
chmod +x lm_concept_learn.sh
algo=knn

for data in yelp indeeda 
do
    for embed in meg-ac meg-pt corel
    do
        for size in 5 10 15 20 25 
        do
            if [ "$embed" = "meg-pt" ]; then
                ./lm_concept_learn.sh $algo $data $embed $size 3072
            else
                ./lm_concept_learn.sh $algo $data $embed $size 768
            fi
        done
    done
done
