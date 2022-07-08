import csv
import gzip
import logging
import sys
import yaml
sys.path.append('../modules')
sys.path.append('../measures/')
sys.path.append('../contextualized/')

import numpy as np
import random
from scipy.spatial.distance import cosine as cosine_distance

from load_data import *
from cos import *
from binary import *
from apd import *
from xlmr import *
#from bert2 import *


def xlmr_baseline():
        # read configurations
        assert os.path.exists('../config/baseline.yaml')
        with open("../config/baseline.yaml", 'r') as config:
            configurations = yaml.safe_load(config)
        config_dict = configurations['xlmr']
        print(config_dict)
        language = config_dict['language']
        type_sentences = config_dict['type_sentences']
        layers = config_dict['layers']
        is_len = config_dict['is_len']
        path_output1 = config_dict['path_output1']
        path_output2 = config_dict['path_output2']
        path_results = config_dict['path_results']
        target_words_path = config_dict['path_targets']

        # target words
        # a hack to produce a target_words lists, standard is to have a list of words at 'target_words_path' as is the case for English
        #target_words = os.listdir('../usage-graph-data/dwug_'+language+'/data/')
        target_words = [w.strip().split('_')[0] for w in open(target_words_path+'/target_words.txt').readlines()][:]
        if target_words_f != []:
            target_words = [w.strip().split('_')[0] for w in target_words_f][:]
        else:
            print('Target word list is empty')
            exit()


        distance_targets_apd = open(path_results+'/apd/distance_targets_xlm_'+language+'_'+type_sentences+'.tsv','w',encoding='utf-8')
        distance_targets_cos = open(path_results+'/cos/distance_targets_xlm_'+language+'_'+type_sentences+'.tsv','w',encoding='utf-8')

        for target_word in target_words:
            uses_corpus1 = []
            uses_corpus2 = []
            print(target_word)
            # load data using the benchmark load_data funciton
            data = load_data(data_path='../usage-graph-data/dwug_'+language+'/',preprocessing='context',lemma=target_word)
            # create two usage corpus sets
            for (lemma,identifier,date,grouping,preprocessing,context_tokenized,indexes_target_token_tokenized,context_lemmatized) in data[0]:
                if grouping == 1:
                    uses_corpus1.append({'lemma':lemma.split('_')[0],'sentence_lemma':context_lemmatized,'index_lemma':indexes_target_token_tokenized,'index_token':indexes_target_token_tokenized,'sentence_token':context_tokenized})
                elif grouping == 2:
                    uses_corpus2.append({'lemma':lemma.split('_')[0],'sentence_lemma':context_lemmatized,'index_lemma':indexes_target_token_tokenized,'index_token':indexes_target_token_tokenized,'sentence_token':context_tokenized})
            # xlmr vectors
            xlmr(uses_corpus1,target_word,language,type_sentences,layers,is_len,path_output1+target_word+'.tsv')
            xlmr(uses_corpus2,target_word,language,type_sentences,layers,is_len,path_output2+target_word+'.tsv')
            # apd and cos distances
            apd_distance = apd(path_output1+target_word+'.tsv',path_output2+target_word+'.tsv')
            distance_targets_apd.write(target_word.encode('utf8','surrogateescape').decode('utf8')+'\t'+str(apd_distance)+'\n')
            cos_distance = cos(path_output1+target_word+'.tsv',path_output2+target_word+'.tsv')
            distance_targets_cos.write(target_word.encode('utf8','surrogateescape').decode('utf8')+'\t'+str(cos_distance)+'\n')

        distance_targets_apd.close()
        distance_targets_cos.close()
        # binary classification
        binary(path_results+'apd/distance_targets_xlm_'+language+'.tsv',path_results+'apd/scores_targets_xlm_'+language+'_'+type_sentences+'.tsv')
        binary(path_results+'cos/distance_targets_xlm_'+language+'.tsv',path_results+'cos/scores_targets_xlm_'+language+'_'+type_sentences+'.tsv')



if __name__ == "__main__":
    xlmr_baseline()
