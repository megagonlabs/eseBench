# meg-kb
KB/Taxonomy of Meg

## Run the code

### Preprocessing and keyword extraction

Run the following to create sentences and phrases from the corpus:
```
cd keyword_extraction
chmod +x ./corpusProcess.sh
$ ./corpusProcess.sh $corpus_name $thread_number $gpu_instance
```

More details are at ``keyword_extraction/README.md``

### Concept Learning
There are two implementations for concept learning, Corel-Concept_Learn and LM-Concept_Learn. 
The Corel-Concept_Learn is forked from https://github.com/teapot123/CoRel.
The LM-Concept_Learn is based on running HDBSCAN on transformer-based embeddings.

To run Corel-Concept_Learn:

```
cd concept_learning
chmod +x ./corel_concept_learn.sh
$ ./corel_concept_learn_initial.sh $corpus_name $thread_number $gpu_instance $topic_file
```

To run LM-Concept_Learn:

```
cd concept_learning
chmod +x ./lm_concept_learn.sh
$ ./lm_concept_learn_initial.sh $corpus_name $thread_number $gpu_instance $embedding_type $clus_algorithm
```

More details are at ``concept_learning/README.md`` 


 