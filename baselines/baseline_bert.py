# the code is largely adopted from
import csv
import gzip
import logging
import sys
import yaml
sys.path.append('../modules')
sys.path.append('../measures/')
sys.path.append('../contextualized/')
sys.path.append('../plots/')
sys.path.append('../')

import numpy as np
import random
from scipy.spatial.distance import cosine as cosine_distance

from load_data import *
from cos import *
from binary import *
from apd import *
from bert import *
from clean_uses import *

def bert_baseline():
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # read configrations
        assert os.path.exists('../config/baseline_bert.yaml')
        with open("../config/baseline_bert.yaml", 'r') as config:
            configurations = yaml.safe_load(config)

        config_dict = configurations['bert']
        #print(config_dict)
        language = config_dict['language']
        type_sentences = config_dict['type_sentences']
        layers = config_dict['layers']
        is_len = config_dict['is_len']
        path_output1 = config_dict['path_output1']
        path_output2 = config_dict['path_output2']
        path_results = config_dict['path_results']
        target_words_path = config_dict['path_targets']

        # target words
        # a hack to produce a target_words lists from dwug data directory, standard is to have a list of words at 'target_words_path' as is the case for English. But there is some encoding issue with some of the German letters so i am using this hack. This is to be fixed in future
        target_words = os.listdir('../usage-graph-data/dwug_'+language+'/data/')
        # Below is the standard way of producing a target_word list, but is commented for German
        '''
        assert os.path.exists(target_words_path+'target_words_'+language+'.txt')
        target_words_f = open(target_words_path+'target_words_'+language+'.txt',encoding='utf-8').readlines()
        if target_words_f != []:
            target_words = [w.strip().encode('utf-8','surrogateescape').decode('utf-8').split('_')[0] for w in target_words_f][:]
        else:
            print('Target word list is empty')
            exit()
        '''

        # output file containing apd and cos distances
        distance_targets_apd = open(path_results+'/apd/distance_targets_bert_'+language+'_'+type_sentences+'.tsv','w',encoding='utf-8')
        distance_targets_apd.write('lemma\tchange_graded\n')

        distance_targets_cos = open(path_results+'/cos/distance_targets_bert_'+language+'_'+type_sentences+'.tsv','w',encoding='utf-8')
        distance_targets_cos.write('lemma\tchange_graded\n')


        for target_word in target_words:
            logging.info("Target Word: "+target_word)
            uses_corpus1 = []
            uses_corpus2 = []
            # load data using the benchmark load_data funciton
            data = load_data(data_path='../usage-graph-data/dwug_'+language+'/',preprocessing='context',lemma=target_word)
            if data == None or data ==  []:
                    print('Usage dataset for the target word is empty')
                    exit()
            # extract usages from time 1 and time 2
            for (lemma,identifier,date,grouping,preprocessing,context_tokenized,indexes_target_token_tokenized,context_lemmatized) in data[0]:
                if grouping == 1:
                    uses_corpus1.append({'lemma':lemma.split('_')[0],'sentence_lemma':context_lemmatized,'index_lemma':indexes_target_token_tokenized,'index_token':indexes_target_token_tokenized,'sentence_token':context_tokenized})
                elif grouping == 2:
                    uses_corpus2.append({'lemma':lemma.split('_')[0],'sentence_lemma':context_lemmatized,'index_lemma':indexes_target_token_tokenized,'index_token':indexes_target_token_tokenized,'sentence_token':context_tokenized})
            # clean uses
            cleaned_uses1 = clean_uses(uses_corpus1,language)
            cleaned_uses2 = clean_uses(uses_corpus2,language)

            # bert vectors
            bert(cleaned_uses1,target_word,language,type_sentences,layers,is_len,path_output1+target_word+'.tsv')
            bert(cleaned_uses2,target_word,language,type_sentences,layers,is_len,path_output2+target_word+'.tsv')

            # compute apd and cos distances
            apd_distance = apd(path_output1+target_word+'.tsv',path_output2+target_word+'.tsv')
            distance_targets_apd.write(target_word.encode('utf8','surrogateescape').decode('utf8')+'\t'+str(apd_distance)+'\n')
            cos_distance = cos(path_output1+target_word+'.tsv',path_output2+target_word+'.tsv')
            distance_targets_cos.write(target_word.encode('utf8','surrogateescape').decode('utf8')+'\t'+str(cos_distance)+'\n')

        distance_targets_apd.close()
        distance_targets_cos.close()
        # binary classification
        #binary(path_results+'apd/distance_targets_bert_'+language+'_'+type_sentences+'.tsv',path_results+'apd/scores_targets_bert_'+language+'_'+type_sentences+'.tsv')
        #binary(path_results+'cos/distance_targets_bert_'+language+'_'+type_sentences+'.tsv',path_results+'cos/scores_targets_bert_'+language+'_'+type_sentences+'.tsv')



if __name__ == "__main__":
    bert_baseline()
