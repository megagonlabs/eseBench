#!/bin/bash
chmod +x lm_concept_learn.sh
algo=kmeans

#yelp indeeda electronics grocery
for data in yelp indeeda
do
    for embed in meg-ac meg-pt corel
    do
        for size in 500 #800 1000 
        do
            if [ "$embed" = "meg-pt" ]; then
                ./lm_concept_learn.sh $algo $data $embed $size 3072 1
            else
                ./lm_concept_learn.sh $algo $data $embed $size 768 1
            fi
        done
    done
done
