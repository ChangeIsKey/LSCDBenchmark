import sys
import pandas as pd
import csv
from collections import defaultdict, Counter
from sklearn import metrics
from scipy import stats
import argparse
import os

def load_data(data_path=None,lemma=None,preprocessing=None):
    #print('in load data')
    assert data_path != None, 'Data path is required'

    if lemma != None:
        #print(lemma)
        lemmas = [l for l in os.listdir(data_path+'/data') if l.split('_')[0] == lemma]
    else:
        #print(os.listdir(data_path+'/data'))
        lemmas = os.listdir(data_path+'/data')
    data = []
    for lemma_dir in lemmas:
        csvfile = data_path + '/data/' + lemma_dir + '/uses.csv'
        df = pd.read_csv(csvfile,delimiter='\t',quoting=csv.QUOTE_NONE, encoding='utf-8')
        #if preprocessing == None: # this is for a revised version of load_data after discussion with the benchmark group in meeting
            #data.append(list(df.to_records(index=False)))
        #else:
            #data.append(list(df.get(['lemma','identifier','date','grouping','context,'context_'+preprocessing]).to_records(index=False)))
        data.append(list(df.get(['lemma','identifier','date','grouping',preprocessing,'context_tokenized','indexes_target_token_tokenized','context_lemmatized']).to_records(index=False)))

    return(data)

if __name__ == "__main__":
    data = load_data(data_path='./usage-graph-data/dwug_en/',preprocessing=None)
    
