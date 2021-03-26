# Concept Learning

There are two implementations for concept learning, Corel-Concept_Learn and LM-Concept_Learn. 
The Corel-Concept_Learn is forked from https://github.com/teapot123/CoRel.
The LM-Concept_Learn is based on running clustering on transformer-based embeddings.

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


## LM-Concept_Learn

There are two different implementations available for getting keyword embeddings:

**Average Context**: the target keyword is masked in a context and sentence embedding of the context is considered as keyword embedding. 
The embeddings are then averaged over different contexts of the keyword in the corpus. 

To obtain these embeddings, use the following command:

```
python compute_keyphrase_embeddings.py -m <model_path> -et ac -d <path_to_intermediate_files> -c <contexts_size>
```

where ```<model_path>``` is either bert-base-uncased or a domain-adapted bert, ```<path_to_intermediate_files>``` is the path to data dir with intermediate files for segmentation, ```ac``` is the embedding type and ```<context_size>``` is the no. of example mentions of a keyword to sample. A sample command would be

```
python compute_keyphrase_embeddings.py -m /home/ubuntu/users/nikita/models/bert_finetuned_lm/indeed_reviews_ques_ans -et ac -d ../../data/indeeda-meg-ac/intermediate -c 750
``` 

or

```
python compute_keyphrase_embeddings.py -m bert-base-uncased -et ac -d ../../data/indeeda-meg-ac/intermediate -c 750
``` 

**Token Representation**: the target keyword is masked in a context and last 4 hidden state representations of the masked token are concatenated to obtain the keyword embedding. The resultant embedding typically has 4*768 dim. 
The embeddings are then averaged over different contexts of the keyword in the corpus.

```
python compute_keyphrase_embeddings.py -m <model_path> -et pt -d <path_to_intermediate_files> -c <contexts_size>
```

An example command would be:

```
python compute_keyphrase_embeddings.py -m /home/ubuntu/users/nikita/models/bert_finetuned_lm/indeed_reviews_ques_ans -et pt -d ../../data/indeeda-meg-pt/intermediate -c 750
``` 

The output embeddings are written to the intermediate data dir under filename ```BERTembed.txt```.

Once the embeddings are obtained, they can be clustered to identify concepts.
There are 3 different clustering algorithms: KMeans, K-NN and Agglomerative Clustering.

To run KMeans:
```
python compute_concept_clusters.py -d <path_to_intermediate_files> -ca kmeans -s <clus_size> -dim <dim> -o <output_filename>
```
where ```<clus_size>``` is the number of clusters, ```<dim>``` is the embedding dim, ```<output_filename>``` is the path to the file for writing the clusters.

To run K-NN:
```
python compute_concept_clusters.py -d <path_to_intermediate_files> -ca knn -s <clus_size> -dim 768 -o <output_filename>
```
where ```<clus_size>``` is the number of neighbors, ```<dim>``` is the embedding dim, ```<output_filename>``` is the path to the file for writing the clusters.

To run Agglomerative Clustering:
```
python compute_concept_clusters.py -d <path_to_intermediate_files> -ca agg -s <clus_size> -dim 768 -o <output_filename>
```
where ```<clus_size>``` is the number of clusters to find, ```<dim>``` is the embedding dim, ```<output_filename>``` is the path to the file for writing the clusters.

## Test Concept Learning Methods

``lm_concept_learn.sh`` autmates the excution of the clustering/concept learning algorithms. It takes six parameters: ``clustering algorithm, dataset path, embedding type, no. of clusters, dimension of embeddings, and min. length of keyphrase used``. You don't need to run this script. There are two other scripts to vary and run the test parameters for different clustering algorithms.
* ``test_lm_kmeans.sh`` varies the datasets, embedding types, and cluster size parameters and runs a trial
* ``test_lm_knn.sh`` varies the datasets, embedding types, and cluster size parameters and runs a trial


  
