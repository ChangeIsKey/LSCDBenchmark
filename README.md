# LSCDBenchmark

  1. [General](#general)
  2. [Data Loading](#data-loading)
  3. [Scoring](#tasks)
  4. [Baselines](#baselines)

    4.1. [Bert](#bert-baseline)

    4.2. [XLMR](#xlmr-baseline)

    4.3. [Random](#random-baseline)

    4.4. [Majority Class](#majority-baseline)

  * [Sub-evaluations](#sub-evaluations)
  * [Datasets](#datasets)
  * [BibTex](#bibtex)


# 1. General

A benchmark for Lexical Semantic Change Detection. It provides.
  1. A data loading function
  2. A scoring script
  3. Baseline systems  

# 2. Data loading
A function to load usage data for one or more lemmas from a given dataset path i.e.

`load_data(data_path=None,lemma=None,preprocessing='lemmatized')`

## 2.1 Parameters:
  + data_path: Absolute path to the data directory
  + lemma: The lemma for which usages are to be loaded from the data directory. If `None` usage data for all lemmas at the    `data_path` are loaded.
  + preprocessing: There are various preprocessed (e.g. lemmatized, tokenized) versions of the usages, this parameter selects a particular version and loads data accordingly.

## 2.2 Output:
  The function returns a list of tuples. Each item in the list corresponds to one data point (i.e. word usage) and has the following values:

  - `(lemma,identifier,date,grouping,preprocessing,context_tokenized,indexes_target_token_tokenized,context_lemmatized)`

# 3. Scoring
The script takes a number of command line arguments and a number of configurable parameters (a yaml file) to compute various evaluation metrics. The metrics are computed for all possible combinations of the values of `filter_label`, `filter_threshold` and `evaluation_type` parameters. These command line and configuration file parameters are briefly explained below.
## 3.1 Command-line arguments:
+ `-p` Absolute path to the file containing predictions.
+ `-g` Absolute path to the file containing gold.
+ `-s`
+ `-a` Absolute path to the file containing annotators agreement scores. These are to be used to filter the data before computing the evaluation metrics.
+ `-o` Absolute path to the directory where output is to be stored.

## 3.2 Configurable parameters (scorer.yaml):
+ `filter_label`
If the data is to be filtered before the evaluation metrics are computed then this parameter contains the name of column on which filtering is to be applied (e.g. kri_full, kri2_full). Can be a single value or a list.
+ `filter_threshold`
The threshold value for the above given `filter_lable`. Can be a single value or a list.  
+ `evaluation_type`
One of the evaluation types i.e. `change_binary` or `change_graded`.
+ `plot`
A binary parameter which can be set to `true` if the plots are .......

An example version of the `scorer.yaml` is as below:

```
evaluation_type:
  - change_binary
filter_label:
  - kri_full
  - kri2_full
filter_threshold:
  - 0.1
  - 0.2
plot: 'False'

```

## 3.3 How to run it:
An example run is as:

`python3 scorer.py -g './usage-graph-data/dwug_en/stats/opt/stats_groupings.csv' -p './results/apd/scores_targets.tsv' -s './results/apd/distance_targets.tsv' -agr './usage-graph-data/dwug_en/stats/stats_agreement.csv' -o './output'`

## 3.4 Output
The evaluation results are stored in `results.csv` file at the output path mentioned with the `-o` parameter. An example run output with the above mentioned configuration file is given below:

| evaluation_type |	filter |	threshold |	 spearmanr_correlation |	spearmanr_pvalue |	f1 |	accuracy | precision |	recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|change_binary|	kri_full|	0.1|	--|	--|	0.44|	0.39|	0.32|	0.71|
|change_binary|	kri_full|	0.2|	--|	--|	0.48|	0.41|	0.36|	0.71|
|change_binary|	kri2_full|	0.1|	--|	--|	0.43|	0.41|	0.3|	0.71|
|change_binary|	kri2_full|	0.2|	--|	--|	0.43|	0.41|	0.3|	0.71|


## 3.5 Plotting:
To be discussed: The ploting functionality in this function requires floating point distances while the gold and prediction data is in binary formate. At the moment i am loading the scoring files, and then extracting scores for the predicted data to be passed to the ploting function, is this what we want?  
## 3.6 Example Usage:



# 4. Baselines

## 4.1 General
The benchmark provides a number of baselines systems for baseline evaluations. There is a separate script for each of the baselines, and is placed inside the `LSCDBenchmark/baselines/` directory. Any of the baseline scripts can be run form command line e.g.

`>>> python3 baseline_majority.py`

Each of the baseline system requires a number of configurable parameters. There is a separate configuration file for each system and is placed inside (`LSCDBenchmark/config/`) directory. A brief description of those parameters is given in [Section 4.2](#4.2-configuration-files)

At this stage, the baselines are adapted to the 'DWUG' datasets, and the systems expect a directory called `usage-graph-data` inside the `LSCDBenchmark` directory containing the datasets. It also means that the `usage-graph-data` directory has a directory (e.g. `dwug_en` for English) containing DWUG dataset of the language for which the baseline is to be run. Running any of the baselines systems from command line (e.g. `python3 baseline_bert.py`) will compute the results and store them at the path given in the `path_results` parameter in the configuration file. For `bert` and `xlmr` baseline systems, two types of scores are computed and are stored inside the `/cos/` and `/apd/` within the `path_results` directory. These results correspond to the cosine similarity and average pair-wise distance between the vectors of a given target word from two time spans. The results of random and majority baseline systems are stored in `rand/` and `/majority/` directories respectively within the `path_results` directory.

## 4.2 Quick Start
1. Download (and unzip) or clone the repository.
2. Go into the  `/LSCDBenchmark/baselines/` directory.
2. Make sure the required libraries are installed (requirements.txt). The systems have been test on MacOS with Python 3.9.10.
3. Make adjustments in the configuration file (see [Section 4.3](#4.3-configuration-files) for details on various configuration parameters) for the baseline system that you are interested to run.
4. Create a directory called `usage-graph-data` inside the 'LSCDBenchmark' directory and download <a href=https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/wugs/> dwug dataset</a> for the language that you are interested in.
5. Create a file named `target_words_[language].txt` (with a trailing language code e.g. `target_words_en.txt` for English) which contains the list of target words (one word per line) and put it at the path mentioned in the `path_targets` configuration parameter.  
6. Run the baseline script (e.g. `>>> python3 baseline_bert.py.`).
7. The results will be stored at  `path_results` given in the configuration file.  

### 4.3 Configuration Files

#### 4.3.1 Bert-Baseline (baseline_bert.yaml)

+ language: The language code (e.g. 'en', 'sv'). The code is used to load data and save results in files with a trailing language code in the file names.
+ type_sentences: The preprocessing type (either 'lemma' or 'toklem'). In case of 'lemma' the lemmatized version of the context sentence is used before computing the embeddins, while in the case of 'toklem' the tokenized version of the context and lemmatized version of the target word is used.  
+ layers: e.g. 1+12
+ is_len: e.g. False
+ f2:
+ max_samples:
+ path_output1: Path to the dirctory where the bert vectors for corpus1 are to be stored (e.g. `./output/vectors_bert_corpus1/` )
+ path_output2: Path to the dirctory where the bert vectors for corpus2 are to be stored (e.g. `./output/vectors_bert_corpus2/``
+ path_results: Path to the diretor where results are to be stored (e.g. `./results/`)
+ path_targets: Path to the directory where target words are to be found (e.g. `./targets/`)

#### 4.3.2 XMLR-Baseline (baseline_xlmr.yaml)

+ language: The language code (e.g. 'en', 'sv')
+ type_sentences: i.e. lemma
+ layers: e.g. 1+12
+ is_len: e.g. False
+ f2:
+ max_samples:
+ path_output1: Path to the dirctory where the bert vectors for corpus1 are to be stored (e.g. `./output/vectors_xlmr_corpus1/` )
+ path_output2: Path to the dirctory where the bert vectors for corpus2 are to be stored (e.g. `./output/vectors_xlmr_corpus2/``
+ path_results: Path to the diretor where results are to be stored (e.g. `./results/`)
+ path_targets: Path to the directory where target words are to be found (e.g. `./targets/`)

Based on the above listed parameters, the script loads usages (from two different time-spans) of a list of target words using the load_data function, computes their bert/xlm vectors, compute cos/apd distances between vectors from two time spans. The distances then are converted to binary labels using a threshold. The final results are stored at the `path_results`.

#### 4.3.3 Random-Baseline (baseline_random.yaml)

+ language: The language code (e.g. 'en', 'sv')
+ path_results: Path to the diretor where results are to be stored (e.g. `./results/`)
+ path_targets: Path to the directory whrer target words are to be found (e.g. `./targets/`)
+ is_rel:

#### 4.3.4 Majority Calass Baseline (baseline_majority.yaml)

+ language: The language code (e.g. 'en', 'sv')
+ path_results: Path to the diretor where results are to be stored (e.g. `./results/`)
+ path_targets: Path to the directory whrer target words are to be found (e.g. `./targets/`)
+ path_data: Path to the directory where data is to be found from which the majority class labels are to be computed.

##### TODO:

Priority:
- [ ] extraction of sense description labels is done more elegantly in WSI-summer benchmark: semeeval_to_bts_rnc_convert.py
- [ ] use hydra or not?
- [ ] implement three scenarios of use (see picture I sent)

- [ ] ===Standard Split===
- [ ] Datasets should provide: dataset1/dev.data, dataset1/eval.data
- [ ] All downloaded datasets should be converted to the same structure
- [ ] converted data should be split into development set and training set
- [ ] Include MCL-WiC, binary labels vs scaled labels

#### Tasks

lemma level:
- binary change (sense loss or gain)
- graded change 1 (divergence of word sense distribution)
- graded change 2 (compare metric)
- clustering evaluation (already implemented)

usage level:
- novel sense identification

##### TODO:
- Accuracy and correlation on WiC
- Later WSI
-

#### Metrics

|Name | Code | Applicability | Comment |
| --- | --- | --- | --- |
| Spearman correlation | `evaluation/spr.py` | DURel, SURel, SemCor LSC, SemEval*, Ru\* | - outputs rho (column 3) and p-value (column 4) |
| Average Precision | `evaluation/ap.py` | SemCor LSC, SemEval*, DIACR-Ita | - outputs AP (column 3) and random baseline (column 4) |
| F1/accuracy | | | |

##### TODO:
- Recall and Precision

#### Sub-Evaluations

- POS
- hard binary changes (which are not also graded changes)
- hard graded changes (which are not also binary changes)
- extended data sets
- very clean cases

##### TODO:
- Nothing currently

#### Datasets

| Dataset | Language | Corpus 1 | Corpus 2 | Download | Comment |
| --- | --- | --- | --- | --- | --- |
| DURel | German | DTA18 | DTA19  | [Dataset](https://www.ims.uni-stuttgart.de/data/durel), [Corpora](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/wocc) | - version from Schlechtweg et al. (2019) at `testsets/durel/` |
| SURel | German | SDEWAC | COOK | [Dataset](https://www.ims.uni-stuttgart.de/data/surel), [Corpora](https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/wocc) | - version from Schlechtweg et al. (2019) at `testsets/surel/` |
| SemCor LSC | English | SEMCOR1 | SEMCOR2 | [Dataset](https://www.ims.uni-stuttgart.de/data/lsc-simul), [Corpora](https://www.ims.uni-stuttgart.de/data/lsc-simul) | |
| SemEval Eng | English | CCOHA 1810-1860 | CCOHA 1960-2010 | [Dataset](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd), [Corpora](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd) | |
| SemEval Ger | German | DTA 1800-1899 | BZND 1946-1990 | [Dataset](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd), [Corpora](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd) | |
| SemEval Lat | Latin | LatinISE -200-0 | LatinISE 0-2000 | [Dataset](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd), [Corpora](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd) | |
| SemEval Swe | Swedish | Kubhist2 1790-1830 | Kubhist2 1895-1903 | [Dataset](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd), [Corpora](https://www.ims.uni-stuttgart.de/data/sem-eval-ulscd) | |
| RuSemShift1 | Russian | RNC 1682-1916 | RNC 1918-1990 | [Dataset](https://github.com/juliarodina/RuSemShift), [Corpora](https://rusvectores.org/static/corpora/) | |
| RuSemShift2 | Russian | RNC 1918-1990 | RNC 1991-2016 | [Dataset](https://github.com/juliarodina/RuSemShift), [Corpora](https://rusvectores.org/static/corpora/) | |
| RuShiftEval1 | Russian | RNC 1682-1916 | RNC 1918-1990 | [Dataset](https://github.com/akutuzov/rushifteval_public), [Corpora](https://rusvectores.org/static/corpora/) | |
| RuShiftEval2 | Russian | RNC 1918-1990 | RNC 1991-2016 | [Dataset](https://github.com/akutuzov/rushifteval_public), [Corpora](https://rusvectores.org/static/corpora/) | |
| RuShiftEval3 | Russian | RNC 1682-1916 | RNC 1991-2016 | [Dataset](https://github.com/akutuzov/rushifteval_public), [Corpora](https://rusvectores.org/static/corpora/) | |
| DIACR-Ita | Italian | Unità 1945-1970 | Unità 1990-2014 | [Dataset](https://github.com/diacr-ita/data/tree/master/test), [Corpora](https://github.com/swapUniba/unita/) | |
| other Italian data set | | | | | |

##### TODO:
- Nothing

#### Leaderboard

#### Pre-trained models?

Compare [here](https://sapienzanlp.github.io/xl-wsd/)

BibTex
--------

```
```


#### TODO for Serge:

- [ ] Clean Up existing code
- [ ] Script for downloading all datasets
- [ ] Implementing of [Tasks](#Tasks)
- [ ] Cleaned Versions of Datasets
- [ ] ...

Notes:
> What structure/functions do we need?
> Load Function,
