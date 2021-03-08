## Step 1: Corpus pre-processing

You need to first create a data folder with dataset name `$DATA` at the project root, and then put your raw text corpus (each line represents a single document) under the `$DATA/source/` folder. 

```
data/$DATA
└── source
    └── corpus.txt
```
Each line in corpus.txt represents a document. 

Finally, you can run the pre-processing pipeline by typing the following command:

```
$ chmod +x ./corpusProcess_new.sh
$ ./corpusProcess_new.sh $corpus_name $thread_number $gpu_instance
```

### Example usage

For example, if your corpus_name is called "wiki" and you want preprocess the data using 8 threads, you can first
put the raw text corpus in "../../data/wiki/source/corpus.txt". Then, you can type the following command:

```
$ ./corpusProcess_new.sh wiki 8
```

The above pipeline will output files, organized as follows:

```
data/$DATA
└── intermediate
    └── sentences.json: documents with entity phrases and noun-chunks
    └── segmentation.txt: the highlighted phrases will be enclosed by the phrase tags (e.g., <phrase>data mining</phrase>).
    └── AutoPhrase_multi-words.txt: the sub-ranked list for multi-word phrases only.
    └── AutoPhrase_single-word.txt: the sub-ranked list for single-word phrases only.
    └── sent_segmentation.txt: sentence-wise the highlighted phrases will be enclosed by the phrase tags (e.g., <phrase>data mining</phrase>).
    └── sentences.json: documents with entity phrases and noun-chunks
	
```






