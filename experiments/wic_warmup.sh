#!/bin/bash

export HYDRA_FULL_ERROR=1
export PROJECT=/projekte/cik/users/andres/LSCDBenchmark
pushd "${PROJECT}"

python main.py -m \
    dataset=dwug_de_210 \
    dataset.exclude_annotators=[] \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=german,none \
    dataset.split=dev \
    task=wic \
    ~evaluation \
    task/wic@task.model=contextual_embedder \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/dist_fn@task.model.similarity_metric=dot \
    task.model.subword_aggregation=average \
    task.model.layers=[23] \
    task.model.layer_aggregation=average \
    task/wic/normalization@task.model.normalization=none \
    task/wic/scaler@task.model.scaler=none

popd

