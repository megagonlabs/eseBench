from multiprocessing import Pool
import os
import sys


def parsing(suffix):
    corpus_name = suffix[0]
    real_suffix = suffix[1]

    keywords_path = "../../data/{}/intermediate/AutoPhrase_single-word.txt"
    keyphrases_path = "../../data/{}/intermediate/AutoPhrase_multi-words.txt"
    input_path = "../../data/{}/intermediate/subcorpus-{}".format(corpus_name, real_suffix)
    output_path = "../../data/{}/intermediate/sentences.json-{}".format(corpus_name, real_suffix)
    cmd = "python3 annotateNLPFeature.py -corpusName {} -input_path {} -output_path {} -real_suffix {} -single_word_vocab {} -multi_word_vocab {}".format(corpus_name, input_path, output_path, real_suffix, keywords_path, keyphrases_path)
    os.system(cmd)


if __name__ == '__main__':
    # python3 multiprocess_annotateNLPFeature.py $DATA $THREAD
    corpus_name = sys.argv[1]
    number_of_processes = int(sys.argv[2])
    suffix_list = []
    for fileName in os.listdir('../../data/{}/intermediate/'.format(corpus_name)):
        if fileName.startswith("subcorpus-"):
            suffix_list.append((corpus_name, fileName[len("subcorpus-"):]))

    p = Pool(number_of_processes)
    p.map(parsing, suffix_list)
