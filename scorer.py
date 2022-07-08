import sys, os, re
import pandas as pd
import csv
from collections import defaultdict, Counter
from sklearn import metrics
from scipy import stats
import argparse
import itertools
import yaml

pd.set_option('mode.chained_assignment', None)

# command line arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gold", help = "absolute path to the file containing gold data",required=True)
parser.add_argument("-p", "--predictions", help = "absolute path to the file containg predictions",required=True)
parser.add_argument("-s", "--scores", help = "absolute path to the file containg prediction scores")
parser.add_argument("-agr", "--agreement", help = "absolute path to the file containing agreement statistics")
parser.add_argument("-o", "--out", help = "absolute path to the output folder",required=True)


args = parser.parse_args()
gold_file = args.gold
predictions_file = args.predictions
predictions_scores_file = args.scores
output_dir = args.out
computed_scores = []

def compute_score(gold,predictions,evaltype,filter_type,filter_threshold):
    results = defaultdict(lambda: 0.0)
    results['evaluation_type'] = evaltype
    results['filter'] = filter_type
    results['threshold'] = filter_threshold
    if evaltype == 'change_graded':
        #results[evaltype] = [('spearmanr',str(stats.spearmanr(gold,predictions)).strip('SpearmanrResult'))]
        print(stats.spearmanr(gold,predictions))
        spr_result = re.findall("\d+\.\d+", str(stats.spearmanr(gold,predictions)))
        #corelation,pvalue = str(stats.spearmanr(gold,predictions)).strip('SpearmanrResult')
        results['spearmanr_correlation'] = round(float(spr_result[0]),2)
        results['spearmanr_pvalue'] = round(float(spr_result[1]),2)
        results['f1'] = '--'
        results['accuracy'] = '--'
        results['precision'] = '--'
        results['recall'] = '--'

    else:

        results['spearmanr_correlation'] = '--'
        results['spearmanr_pvalue'] = '--'
        results['f1'] = round(float(metrics.f1_score(gold,predictions)),2)
        results['accuracy'] = round(float(metrics.accuracy_score(gold,predictions)),2)
        results['precision'] = round(float(metrics.precision_score(gold,predictions)),2)
        results['recall'] = round(float(metrics.recall_score(gold,predictions)),2)

    return results

def plot_precision_recall(filtered_pred,y_test,pred_score_path,fl,thr):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from funcsigs import signature
    from sklearn.metrics import average_precision_score

    # y_scores
    pred_scores = pd.read_csv(pred_score_path,delimiter='\t',quoting=csv.QUOTE_NONE)
    y_score = pred_scores.loc[pred_scores['lemma'].isin(filtered_pred['lemma'])].get(['distance'])


    precision, recall, threshold = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    #print(precision,recall,threshold)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f},{fl}={thr}'.format(
          average_precision,fl=fl,thr=thr))
    plt.savefig('./results/plots/'+fl+'-'+str(thr)+'.png')
    plt.show()


# load golde
golddata = pd.read_csv(gold_file,delimiter='\t',quoting=csv.QUOTE_NONE)
golddata['lemma'] = golddata['lemma'].str.replace('_(\w+)','',regex=True) # remove trailing pos tag with target words
golddata = golddata.sort_values(by=['lemma'])
#print(golddata['lemma'])

# load predictions
preddata = pd.read_csv(predictions_file,delimiter='\t',quoting=csv.QUOTE_NONE)
preddata['lemma'] = preddata['lemma'].str.replace('_(\w+)','',regex=True) # remove trailing pos tag with target words
preddata = preddata.sort_values(by=['lemma'])



assert os.path.exists('./config/scorer.yaml')

with open("./config/scorer.yaml", 'r') as config:
    configurations = yaml.safe_load(config)
config_dict = configurations['dwug_format']
print(config_dict)

if config_dict['evaluation_type']:
    evaltypes = config_dict['evaluation_type']
else:
    evaltypes = ["change_binary","change_binary_gain","change_binary_loss","change_graded"]

for evaltype in evaltypes: # iterate for every evaluation type
    # filter data based on filter threshold and label given in configuration file
    if config_dict['filter_threshold'] and config_dict['filter_label']:
        if args.agreement:
            stats_agreement = pd.read_csv(args.agreement,delimiter='\t',quoting=csv.QUOTE_NONE)

            for (fl,thr) in itertools.product(config_dict['filter_label'],config_dict['filter_threshold']): # iterate for every possible combination of filter lable and threshold
                filtered_data = stats_agreement.loc[(stats_agreement[fl] >= float(thr))]
                filtered_data['data'] = filtered_data['data'].str.replace('_(\w+)','',regex=True)
                gold = golddata.loc[golddata['lemma'].isin(filtered_data['data'])].get([evaltype])
                pred = preddata.loc[preddata['lemma'].isin(filtered_data['data'])].get([evaltype])
                filtered_pred = preddata.loc[preddata['lemma'].isin(filtered_data['data'])]

                score = compute_score(gold,pred,evaltype,fl,thr)
                computed_scores.append(score)
                if evaltype in ['change_binary'] and config_dict['plot'] == 'True':
                    print('plotting')
                    plot_precision_recall(filtered_pred,gold,predictions_scores_file,fl,thr)

        else:
            print('-agr/--agreement argument is required')
            exit()
    else: # no filtering
        gold = golddata.get([evaltype])
        pred = preddata.get([evaltype])
        #print(len(gold),len(pred))
        score = compute_score(gold,pred,evaltype,'no-cleaning','no-cleaning')
        computed_scores.append(score)
        if evaltype in ['change_binary'] and config_dict['plot'] == 'True':
            print('plotting')
            plot_precision_recall(preddata,gold,predictions_scores_file,'no-cleaning','no-cleaning')



# output results in the output directory
with open(output_dir + '/results.csv', 'w') as f:
    w = csv.DictWriter(f, computed_scores[0].keys(), delimiter='\t', quoting = csv.QUOTE_NONE, quotechar='')
    w.writeheader()
    w.writerows(computed_scores)
