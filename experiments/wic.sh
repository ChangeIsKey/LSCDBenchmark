#!/bin/bash

export HYDRA_FULL_ERROR=1
export PROJECT=/projekte/cik/users/andres/LSCDBenchmark
pushd "${PROJECT}"

# Graded WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset.split=dev,dev1,dev2 \
    task=wic \
    evaluation=wic \
    evaluation/metric=pearson,spearman \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/dist_fn@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],[23,22,21,20],range(0,24),[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none


# Binary WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset.split=dev,dev1,dev2 \
    task=wic \
    evaluation=binary_wic \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/dist_fn@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],range(20,24),range(0,24),[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none

# Graded WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset.split=dev,dev1,dev2 \
    task=wic \
    evaluation=wic \
    evaluation/metric=pearson,spearman \
    task/wic@task.model=deepmistake \
    task.model.ckpt=first_concat,mean_dist_l1ndotn_CE,mean_dist_l1ndotn_MSE,WIC_DWUG+XLWSD,WIC_RSS,WIC+RSS+DWUG+XLWSD \
    task/wic/scaler@task.model.scaler=none

# Binary WIC, DWUG DE, DeepMistake
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset.split=dev,dev1,dev2 \
    task=wic \
    evaluation=binary_wic \
    task/wic@task.model=deepmistake \
    task.model.ckpt=first_concat,mean_dist_l1ndotn_CE,mean_dist_l1ndotn_MSE,WIC_DWUG+XLWSD,WIC_RSS,WIC+RSS+DWUG+XLWSD \
    task/wic/scaler@task.model.scaler=none
popd
