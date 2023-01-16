#!/bin/bash

export HYDRA_FULL_ERROR=1
export PROJECT=/projekte/cik/users/andres/LSCDBenchmark
pushd "${PROJECT}"
source .env
source venv/bin/activate

# Graded WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=none \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=wic \
    evaluation/metric=pearson,spearman \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/metric@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],[23,22,21,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none,minmax \
    hydra.sweeper.max_batch_size=5

python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,tokenization \
    dataset/spelling_normalization=german \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=wic \
    evaluation/metric=pearson,spearman \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/metric@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],[23,22,21,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none,minmax \
    hydra.sweeper.max_batch_size=500

# Binary WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=none \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=binary_wic \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/metric@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],[23,22,21,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none,minmax \
    hydra.sweeper.max_batch_size=500

python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,tokenization \
    dataset/spelling_normalization=german \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=binary_wic \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/metric@task.model.similarity_metric=dot,euclidean,manhattan \
    task.model.subword_aggregation=average,sum,first,last,max \
    task.model.layers=[23],[0,23],[23,22,21,20],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],[12],[14],[8] \
    task.model.layer_aggregation=average,concat,sum \
    task/wic/normalization@task.model.normalization=l1,l2,none \
    task/wic/scaler@task.model.scaler=none,minmax \
    hydra.sweeper.max_batch_size=5

# Graded WIC, DWUG DE, BERT+XLMR
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=german,none \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=wic \
    evaluation/metric=pearson,spearman \
    task/wic@task.model=deepmistake \
    task/wic/scaler@task.model.scaler=none,minmax \
    task.model.ckpt=first_concat,mean_dist_l1ndotn_CE,mean_dist_l1ndotn_MSE,WIC_DWUG+XLWSD,WIC_RSS,WIC+RSS+DWUG+XLWSD

# Binary WIC, DWUG DE, DeepMistake
python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[],[annotator0],[annotator1],[annotator2],[annotator3],[annotator4],[annotator5],[annotator6],[annotator7] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=german,none \
    dataset/split=dev,dev1,dev2 \
    task=wic \
    evaluation=binary_wic \
    task/wic@task.model=deepmistake \
    task/wic/scaler@task.model.scaler=none,minmax \
    task.model.ckpt=first_concat,mean_dist_l1ndotn_CE,mean_dist_l1ndotn_MSE,WIC_DWUG+XLWSD,WIC_RSS,WIC+RSS+DWUG+XLWSD
popd
