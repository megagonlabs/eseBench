"""
__author__: Dongming Lei, Jiaming Shen
__description__: Run SpaCy tool on AutoPhrase output and filter non-NP quality phrases.
    Please have SpaCy Installed: https://spacy.io/usage/
        `pip install -U spacy`
        `python -m spacy download en`
    Input: 1) segmentation.txt, each line is a single document
    Output: 1) a tmp sentences.json.spacy
__latest_updates__: 05/07/2018
"""

import sys
import time
import json
import pandas as pd
import re
from tqdm import tqdm
from collections import deque
import spacy
from spacy.symbols import ORTH, LEMMA, POS, TAG
import mmap
import argparse

DEBUG = True

# INIT SpaCy
nlp = spacy.load('en_core_web_sm')
start_phrase = [{ORTH: u'<phrase>', LEMMA: u'', POS: u'START_PHRASE', TAG: u'START_PHRASE'}]
end_phrase = [{ORTH: u'</phrase>', LEMMA: u'', POS: u'END_PHRASE', TAG: u'END_PHRASE'}]
nlp.tokenizer.add_special_case(u'<phrase>', start_phrase)
nlp.tokenizer.add_special_case(u'</phrase>', end_phrase)

p2tok_list = {}  # global cache of phrase to token


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # add space before and after <phrase> tags
    text = re.sub(r"<phrase>", " <phrase> ", text)
    text = re.sub(r"</phrase>", " </phrase> ", text)
    # text = re.sub(r"<phrase>", " ", text)
    # text = re.sub(r"</phrase>", " ", text)
    # add space before and after special characters
    text = re.sub(r"([.,!:?()])", r" \1 ", text)
    # replace multiple continuous whitespace with a single one
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("-", " ")

    return text


def find(haystack, needle):
    """Return the index at which the sequence needle appears in the
    sequence haystack, or -1 if it is not found, using the Boyer-
    Moore-Horspool algorithm. The elements of needle and haystack must
    be hashable.

    >>> find([1, 1, 2], [1, 2])
    1

    """
    h = len(haystack)
    n = len(needle)
    skip = {needle[i]: n - i - 1 for i in range(n - 1)}
    i = n - 1
    while i < h:
        for j in range(n):
            if haystack[i - j].lower() != needle[-j - 1]:
                i += skip.get(haystack[i].lower(), n)
                break
        else:
            return i - n + 1
    return -1


def obtain_p_tokens(p):
    '''
    :param p: a phrase string
    :return: a list of token text
    '''

    if p in p2tok_list:
        return p2tok_list[p]
    else:
        p_tokens = [tok.text for tok in nlp(p)]
        p2tok_list[p] = p_tokens
        return p_tokens


def merge_entity_mentions(entity_mentions, tokens, single_words, multi_words):
    if len(entity_mentions) < 2:
        return entity_mentions
    sorted_entity_mentions = sorted(entity_mentions, key=lambda x: (x['start'], x['end']))
    already_merged_mentions = set()
    merged_mentions = []
    for i in range(len(sorted_entity_mentions) - 1):
        m1 = sorted_entity_mentions[i]
        m2 = sorted_entity_mentions[i+1]
        if str(m1) in already_merged_mentions:
            continue
        if (m1['start'] <= m2['end']) & ((m2['start'] - 1) <= m1['end']):  # overlapping spans
            # print('overlapping span')
            merged_span = tokens[m1['start']: m2['end'] + 1]
            merged_text = ' '.join(merged_span).lower()
            # print('merged_text: {}'.format(merged_text))
            if merged_text in multi_words:
                ent = {"text": merged_text, "start": m1['start'], "end": m2['end'], "type": "phrase"}
                # print('merged_text in multi words: {}'.format(ent))
                merged_mentions.append(ent)
                already_merged_mentions.add(str(m1))
                already_merged_mentions.add(str(m2))
        else:
            merged_mentions.append(m1)
            already_merged_mentions.add(str(m1))
    # any remaining unused mentions
    for m in sorted_entity_mentions:
        if str(m) not in already_merged_mentions:
            merged_mentions.append(m)
    return merged_mentions


def process_one_doc(article, articleId, single_words, multi_words):
    result = []
    phrases = []
    output_token_list = []

    # go over once
    article = clean_text(article)
    q = deque()
    IN_PHRASE_FLAG = False
    for token in article.split(" "):
        if token == "<phrase>":
            IN_PHRASE_FLAG = True
        elif token == "</phrase>":
            current_phrase_list = []
            while (len(q) != 0):
                current_phrase_list.append(q.popleft())
            phrases.append(" ".join(current_phrase_list).lower())
            IN_PHRASE_FLAG = False
        else:
            if IN_PHRASE_FLAG:  # in the middle of a phrase, push the token into queue
                q.append(token)

            ## put all the token information into the output fields
            output_token_list.append(token)

    text = " ".join(output_token_list)

    doc = nlp(text)

    sentId = 0
    for sent in doc.sents:  # seems to me doc.sents is just to separate a sentence into several parts (according to ':')
        NPs = []
        pos = []

        lemmas = []
        deps = []

        tokens = []
        for s in sent.noun_chunks:
            NPs.append(s)

        # get pos tag and dependencies
        for token in sent:
            tokens.append(token.text)
            pos.append(token.tag_)
            lemmas.append(token.lemma_)
            deps.append(token.dep_)

        entityMentions = []
        # For each quality phrase, check if it's NP
        for p in phrases:  # phrases are always in lower case.
            for np in NPs:
                # find if p is a substring of np
                if np.text.lower().find(p) != -1:
                    sent_offset = sent.start

                    # tmp = nlp(p)
                    p_tokens = obtain_p_tokens(p)  # Just to partition p into several tokens.
                    # p_tokens = [tok.text for tok in tmp]

                    offset = find(tokens[np.start - sent_offset:np.end - sent_offset], p_tokens)
                    if offset == -1:
                        # SKIP THIS AS THIS IS NOT EXACTLY MATCH
                        continue

                    start_offset = np.start + offset - sent_offset
                    ent = {"text": " ".join(p_tokens), "start": start_offset,
                           # "end": start_offset + len(p.split(" ")) - 1,
                           "end": start_offset + len(p_tokens) - 1, "type": "phrase"}

                    # sanity check
                    if ent["text"] != " ".join(x.lower() for x in tokens[ent["start"]:ent["end"] + 1]):
                        print("NOT MATCH", p, " ".join(tokens[ent["start"]:ent["end"] + 1]))
                        print("SENT", " ".join(tokens))
                        print("SENT2", sent.text)

                    ## TODO: check why there are duplicates in entityMentions
                    entityMentions.append(ent)
        merged_entity_mentions = merge_entity_mentions(entityMentions, tokens, single_words, multi_words)
        res = {"articleId": articleId, "sentId": sentId, "tokens": tokens, "pos": pos, "lemma": lemmas, "dep": deps,
               "raw_entityMentions": entityMentions,
               "entityMentions": merged_entity_mentions,
               "np_chunks": [{"text": t.text, "start": t.start - sent.start, "end": t.end - sent.start - 1} for t in
                             NPs]}
        result.append(res)
        sentId += 1

    return result


def process_corpus(input_path, output_path, real_suffix, single_word_vocab_path, multi_word_vocab_path):
    start = time.time()
    single_words = pd.read_csv(single_word_vocab_path, header=None, names=['prob', 'phrase'], delimiter='\t')['phrase'].tolist()
    multi_words = pd.read_csv(multi_word_vocab_path, header=None, names=['prob', 'phrase'], delimiter='\t')
    multi_words = multi_words[multi_words['prob'] >= 0.45]['phrase'].tolist()
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for cnt, line in tqdm(enumerate(fin), total=get_num_lines(input_path)):
            line = line.strip()
            # try:
            article_result = process_one_doc(line, "{}-{}".format(real_suffix, cnt), single_words, multi_words)
            for sent in article_result:
                json.dump(sent, fout)
                fout.write("\n")
            # except:
            #     print("exception")
    end = time.time()
    print("Finish NLP processing, using time %s (second)" % (end - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='main.py', description='')
    parser.add_argument('-corpusName', required=False, default='sample_dataset', help='corpusName: sample_dataset or sample_wiki or wiki')
    parser.add_argument('-input_path', required=False, default='data/sample_dataset/intermediate/segmentation.txt', help='input_path')
    parser.add_argument('-output_path', required=False, default='data/sample_dataset/intermediate/sentences.json.spacy', help='output_path')
    parser.add_argument('-real_suffix', required=False, default="aa", help='real_suffix: used to prepend for articleID')  # used to prepend for articleID
    parser.add_argument('-single_word_vocab', required=False, default='data/sample_dataset/intermediate/AutoPhrase_single-word.txt')
    parser.add_argument('-multi_word_vocab', required=False,
                        default='data/sample_dataset/intermediate/AutoPhrase_multi-words.txt')
    args = parser.parse_args()
    process_corpus(args.input_path, args.output_path, args.real_suffix, args.single_word_vocab, args.multi_word_vocab)
