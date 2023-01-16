#!/bin/bash

export HYDRA_FULL_ERROR=1
export PROJECT=/projekte/cik/users/andres/LSCDBenchmark
pushd "${PROJECT}"
source .env
source venv/bin/activate

python main.py -m \
    dataset=dwug_de_210 \
    dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
    dataset/spelling_normalization=german,none \
    dataset/split=full \
    task=wic \
    evaluation=none \
    task/wic@task.model=contextual_embedder \
    task.model.encode_only=true \
    task.model.ckpt=deepset/gbert-large,xlm-roberta-large \
    task.model.gpu=0 \
    task/wic/dist_fn@task.model.similarity_metric=dot \
    task.model.subword_aggregation=first \
    task.model.layers=[2] \
    task.model.layer_aggregation=sum

# python main.py -m \
#     dataset=dwug_en_200 \
#     dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
#     dataset/spelling_normalization=english,none \
#     ++dataset.split=full \
#     task=wic \
#     evaluation=none \
#     task/wic@task.model=contextual_embedder \
#     task.model.encode_only=true \
#     task.model.ckpt=bert-large-uncased,xlm-roberta-large \
#     task.model.gpu=0 \
#     task/wic/dist_fn@task.model.similarity_metric=dot \
#     task.model.subword_aggregation=first \
#     task.model.layers=[12] \
#     task.model.layer_aggregation=sum \
#     task/wic/normalization@task.model.normalization=none \
#     task/wic/scaler@task.model.scaler=none

# python main.py -m \
#     dataset=dwug_sv_200 \
#     dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
#     dataset/spelling_normalization=swedish,none \
#     ++dataset.split=full \
#     task=wic \
#     evaluation=none \
#     task/wic@task.model=contextual_embedder \
#     task.model.encode_only=true \
#     task.model.ckpt=KB/bert-base-swedish-cased,xlm-roberta-large \
#     task.model.gpu=0 \
#     task/wic/dist_fn@task.model.similarity_metric=dot \
#     task.model.subword_aggregation=first \
#     task.model.layers=[12] \
#     task.model.layer_aggregation=sum \
#     task/wic/normalization@task.model.normalization=none \
#     task/wic/scaler@task.model.scaler=none

# python main.py -m \
#     dataset=dwug_es_400 \
#     dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
#     dataset/spelling_normalization=none \
#     ++dataset.split=full \
#     task=wic \
#     evaluation=none \
#     task/wic@task.model=contextual_embedder \
#     task.model.encode_only=true \
#     task.model.ckpt=dccuchile/bert-base-spanish-wwm-uncased,xlm-roberta-large \
#     task.model.gpu=0 \
#     task/wic/dist_fn@task.model.similarity_metric=dot \
#     task.model.subword_aggregation=first \
#     task.model.layers=[12] \
#     task.model.layer_aggregation=sum \
#     task/wic/normalization@task.model.normalization=none \
#     task/wic/scaler@task.model.scaler=none

popd

