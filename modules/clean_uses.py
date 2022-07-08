#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import gzip
import logging
import time

from docopt import docopt
from fuzzywuzzy import fuzz


def clean_uses(sentences,language):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("Cleaning Uses".upper())
    logging.info(__file__.upper())
    start_time = time.time()

    #if language == 'ger':
    if language == 'de':
        trans_table = {u'aͤ' : u'ä', u'oͤ' : u'ö', u'uͤ' : u'ü', u'Aͤ' : u'Ä',
                    u'Oͤ' : u'Ö', u'Uͤ' : u'Ü', u'ſ' : u's', u'\ua75b' : u'r',
                    u'm̃' : u'mm', u'æ' : u'ae', u'Æ' : u'Ae',
                    u'göñ' : u'gönn', u'spañ' : u'spann'}
    elif language == 'en':
        trans_table = {u' \'s' : u'\'s',
                    u' n\'t' : u'n\'t', u' \'ve' : u'\'ve', u' \'d' : u'\'d',
                    u' \'re' : u'\'re', u' \'ll' : u'\'ll'}
    elif language == 'sv':
        trans_table = {u' \'s' : u'\'s'}
    else:
        trans_table = {}

    sentences_lemma = []
    sentences_token = []
    index_lemma = []
    index_token =[]

    for sentence in sentences:
        sentences_lemma.append(sentence["sentence_lemma"])
        sentences_token.append(sentence["sentence_token"])

    # Clean sentences
    for i in range(0, len(sentences_lemma)):
        bcleaning = sentences_token[i]

        for key, value in trans_table.items():
            #sentences_lemma[i] = sentences_lemma[i].replace(key, value)
            sentences_token[i] = sentences_token[i].replace(key, value)
        acleaning= sentences_token[i]

    lemma = sentences[0]["lemma"]

    # Find new target_index for lemmatized sentence
    for sentence_lemma in sentences_lemma:
        max_ratio = 0
        for word in sentence_lemma.split():
            ratio = fuzz.ratio(lemma, word.lower())
            if ratio > max_ratio:
                max_ratio = ratio
                index = sentence_lemma.split().index(word)
        index_lemma.append(index)

    # Find new target_index for tokenized sentence
    for sentence_token in sentences_token:
        max_ratio = 0
        for word in sentence_token.split():
            ratio = fuzz.ratio(lemma, word.lower())
            if ratio > max_ratio:
                max_ratio = ratio
                index = sentence_token.split().index(word)
        index_token.append(index)


    cleaned_sentences = []
    for i in range(0, len(sentences_lemma)):
        cleaned_sentences.append({"sentence_lemma":sentences_lemma[i], "sentence_token":sentences_token[i], "index_lemma":index_lemma[i], "index_token":index_token[i], "lemma":lemma})
        #writer.writerow([sentences_lemma[i], sentences_token[i], index_lemma[i], index_token[i], lemma])

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    return(cleaned_sentences)


if __name__ == '__main__':
    main()
