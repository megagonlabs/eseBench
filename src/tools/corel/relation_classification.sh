#dataset directory
dataname=$1
topic=$2
dataset=../../data/${dataname}/intermediate
topic_file=topics_${topic}.txt

python main.py --dataset $dataset --topic_file $topic_file
