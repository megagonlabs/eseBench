# Concept Learning

There are two implementations for concept learning, Corel-Concept_Learn and LM-Concept_Learn. 
The Corel-Concept_Learn is forked from https://github.com/teapot123/CoRel.
The LM-Concept_Learn is based on running HDBSCAN on transformer-based embeddings.

## Corel-Concept_Learn
To run Corel-Concept_Learn:
```
cd concept_learning
chmod +x ./corel_concept_learn.sh
$ ./corel_concept_learn_initial.sh $corpus_name $thread_number $gpu_instance $topic_file
```

An example command is:

```
./corel_concept_learn_initial.sh indeeda 4 1 des
```

This step compiles the source file and trains embedding for concept learning. 
The ```topic_file``` specifies the seed taxonomy. 
If you want to specify your own seed taxonomy, just feel free to create a new file using the format ```topics_{xxx}.txt```. 
Each line starts with a parent node (with the root node being ROOT), and then followed by a tab. 
The children nodes of this parent is appended and separated by space. 
Generated embedding file is stored under ```${corpus_name}```.