# Working Examples

You can create an combination of configurations based on your needs.

## Step 1

Start your basic configurations with as the following code:

```sh
    python main.py -m \
        dataset=dwug_de_210 \
        task=wic \
        evaluation=none \
```

There are options for [dataset](#the-dataset-options), [task](#the-task-options), and [evaluation](#the-evaluation-options).

### The dataset options

- diawug_110
- discowug_110
- durel_300
- dwug_de_210
- dwug_de_sense
- dwug_en_200
- dwug_es_400
- dwug_la_1
- dwug_sv_200
- nordiachange_1
- nordiachange_2
- refwug_110
- rusemshift_1
- rusemshift_2
- rushifteval_1
- rushifteval_2
- rushifteval_3
- surel_300

### The task options

- lscd_binary
- lscd_compare
- lscd_graded
- wic
- wsi

### The evaluation options

- binary_wic
- change_binary
- change_graded
- compare
- none
- wic
- wsi

## Step 2

The bash will return the feedback message to ask you specify more detail infromation. For example, if you choose to use LSCD_binary model, you will be ask to specify `task/lscd_binary@task.model` with the options of `apd_compare_all` and `cos`. If you use LSCD_graded model, you will be ask to specify `task/lscd_graded@task.model` with the options of `apd_compare_all`, `apd_compare_annotated`, `cluster_jsd`, `permutation`, etc.

## Examples

Except for the example combination of configurations in {ref}`usage` page, we provide some more examples for reference.

1. An example, using the German dataset `dwug_de_210`, with model `contextual_embedder` using German BERT as a WiC model and without evaluation setting would be the following:

```sh
    python main.py -m \
        dataset=dwug_de_210 \
        dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
        dataset/spelling_normalization=german,none \
        dataset/split=full \
        task=wic \
        evaluation=none \
        task/wic@task.model=contextual_embedder \
        task.model.ckpt=deepset/gbert-large \
        task.model.gpu=0 \
        task/wic/metric@task.model.similarity_metric=dot
```

2. An example, using the English dataset `dwug_en_210`, with model `contextual_embedder` using BERT as a WiC model and without evaluation setting would be the following:

```sh
    python main.py -m \
        dataset=dwug_en_200 \
        dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
        dataset/spelling_normalization=english,none \
        dataset/split=full \
        task=wsi \
        evaluation=none \
        task/wsi@task.model=cluster_correlation \
        task/wic@task.model.wic=contextual_embedder \
        task.model.wic.ckpt=bert-large-uncased \
        task.model.wic.gpu=0 \
        task/wic/metric@task.model.wic.similarity_metric=dot
```

3. An example, using the Swedish dataset `dwug_sv_200`, with model `contextual_embedder` using German BERT as a WiC model and without evaluation setting would be the following:

```sh
    python main.py -m \
        dataset=dwug_sv_200 \
        dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
        dataset/spelling_normalization=swedish,none \
        dataset/split=full \
        task=lscd_binary \
        evaluation=none \
        task/lscd_binary@task.model=apd_compare_all \
        task/lscd_binary/threshold_fn@task.model.threshold_fn=mean_std \
        task/wic@task.model.graded_model.wic=contextual_embedder \
        task.model.graded_model.wic.ckpt=KB/bert-base-swedish-cased \
        task.model.graded_model.wic.gpu=0 \
        task/wic/metric@task.model.graded_model.wic.similarity_metric=dot
```

4. An example, using the Spenish dataset `dwug_es_400`, with model `contextual_embedder` using BERT as a WiC model and without evaluation setting would be the following:

```sh
    python main.py -m \
        dataset=dwug_es_400 \
        dataset/preprocessing=toklem,raw,tokenization,normalization,lemmatization \
        dataset/spelling_normalization=none \
        dataset/split=full \
        task=lscd_graded \
        evaluation=none \
        task/lscd_graded@task.model=apd_compare_all \
        task/wic@task.model.wic=contextual_embedder \
        task.model.wic.ckpt=dccuchile/bert-base-spanish-wwm-uncased \
        task.model.wic.gpu=0 \
        task/wic/metric@task.model.wic.similarity_metric=dot
```
