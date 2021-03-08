# CoRel: Seed-Guided Topical Taxonomy Construction by Concept Learning and Relation Transferring

## Requirements

* GCC compiler (used to compile the source c file): See the [guide for installing GCC](https://gcc.gnu.org/wiki/InstallingGCC).

## Runng the Code

### Concept Learning for topic nodes
```
./concept_learn_inital.sh $dataset_name $topic_name
```

An example command is:
```
./concept_learn_inital.sh indeeda des
```

Here, ``des`` indicates the VERSION of your topic file which is usually named as ``topics_VERSION.txt``, e.g., topics_des.txt.

### Relation Transferring
```
./relation_classification.sh $dataset_name $topic_name
```
As an example, for indeeda dataset, you can run
```
./relation_classification.sh indeeda des
```
### Concept learning for all nodes
```
./concept_learn_all.sh $dataset_name
```
