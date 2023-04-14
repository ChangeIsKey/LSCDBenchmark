# Installation instructions

To get started, clone the repository, then create a new environment:

```sh
git clone https://github.com/ChangeIsKey/LSCDBenchmark.git

virtualenv env
source env/bin/activate
pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

## Getting Started

`LSCDBenchmark` heavily relies on [Hydra](https://hydra.cc/) for easily configuring experiments.

By running `python main.py`, the tool will guide you towards specifying some of its required parameters. The main parameters are:

- dataset
- evaluation
- task

From the shell, Hydra will ask you to provide values for all these parameters,
and will provide you with a list of options.
Once you select a value for each of these parameters, you might have to input
other, deeply nested required parameters. You can define a script to run your
experiments if you constantly find yourself typing the same command, as these
can get quite verbose.

An example, using the dataset `dwug_de`, with model `apd_compare_all` using BERT
as a WiC model and evaluating against graded change labels would be the
following:

```bash
python main.py \
  dataset=dwug_de_210 \
  dataset/split=dev \
  dataset/spelling_normalization=german \
  dataset/preprocessing=normalization \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task/wic/metric@task.model.wic.similarity_metric=cosine \
  task.model.wic.ckpt=bert-base-german-cased \
  task.model.wic.gpu=0 \
  'dataset.test_on=[abbauen,abdecken,"abgebrüht"]' \
  evaluation=change_graded
```

Here, we chose `contextual_embedder` as a word-in-context model. This model
requires a `ckpt` parameter, which represents any model stored in [Huggingface
Hub](https://huggingface.co/models), like `bert-base-cased`,
`bert-base-uncased`, `xlm-roberta-large`, or
`dccuchile/bert-base-spanish-wwm-cased`.

`contextual_embedder` can also accept a `gpu` parameter. This parameter takes an
integer, and represents the ID of a certain GPU (there might be multiple on a
single machine).


## Running in inference mode

If you don't want to evaluate a model, you can use tilde notation (~) to remove a certain required parameter. For example, to run the previous command without any evaluation, you can run the following:

```bash
python main.py \
  dataset=dwug_de \
  task=lscd_graded \
  task/lscd_graded@task.model=apd_compare_all \
  task/wic@task.model.wic=contextual_embedder \
  task.model.wic.ckpt=bert-base-german-cased \
  ~evaluation
```


## Possible issues
- you may have to adjust the CUDA version specified in requirements.txt to your local requirements for the GPU.