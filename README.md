# meg-kb expansion tool

KB expansion tool that given a corpus and a seed set of entities of user-defined categories, finds more entities that have the same categories.

## Run the code

## Step 1: Corpus pre-processing

You need to first create a data folder with dataset name `$DATA` at the project root, and then put your raw text corpus (each line represents a single document) under the `$DATA/source/` folder. 

```
data/$DATA
└── source
    └── corpus.txt
```
Each line in corpus.txt represents a document. 


## Step 2: Seed Definition

You next need to create a (comma-separated) csv file with the name `seed_aligned_concepts.csv` under the `$DATA` folder to specify the the seed entities. 

```
data/$DATA
└── seed_aligned_concepts.csv
```

Here is what it should look like:

```
=======================================================================================================================================================
| alignedCategoryName              |   unalignedCategoryName |      generalizations |   seedInstances                                                 |
|----------------------------------+-------------------------+---------------------+----------------------------------------------------------------- |
| technology                       |   technology            |                      |"['python', 'sql', 'java', 'html', 'perl', 'javascript', 'php']" |
| programming_language             |   programming language  |                      |"['distributed systems', 'load balancing', 'network monitoring']"|
=======================================================================================================================================================
```

`alignedCategoryName` is a canonical name of the category. `unalignedCategoryName` is the common name of the category used by the pre-trained language model. `generalizations` if any for the category. this field is optional. `seedInstances` is a comman-separated list of seed entities. typically 5-10 entities are sufficient per category. 

## Step 3: Run the tool

Run the following to create keywords as entity candidates from the corpus, learn their embeddings and create a ranked list of entities for each category.
```
cd src
source activate /nfs/users/nikita/.conda/envs/kb_entity_expan
./expand_taxonomy.sh $DATASET_NAME
```

This creates intermediate and output files at: 

```
data/$DATA
└── intermediate
    └── AutoPhrase_single-word.txt: the sub-ranked list for single-word phrases only.
    └── AutoPhrase_multi-words.txt: the sub-ranked list for multi-word phrases only.
    └── sent_segmentation.txt: sentence-wise the highlighted phrases will be enclosed by the phrase tags (e.g., <phrase>data mining</phrase>).
    └── sentences.json: documents with entity phrases and noun-chunks
    └── BERTembed+seeds.txt: embeddings for the keyphrases
    └── ee_mrr_combine_bert_k=200.csv: top-200 predictions based on MRR over corpus embeddings and PLM rankings
    └── ee_concept_knn_k=None.csv: ranked list of entities based on corpus embeddings
    └── ee_LM_bert_k=None.csv: ranked list of entities based on PLM
```
