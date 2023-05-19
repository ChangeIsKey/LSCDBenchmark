# Working Examples

You can create an combination of configurations based on your needs.

## Step 1

Start your basic configurations with as the following example command line:

```sh
python main.py -m \
    dataset=dwug_de_210 \
    task=wic \
    evaluation=none \
```

There are options for dataset, task, and evaluation.

````{tabs}
```{tab} dataset

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
```
```{tab} task

- lscd_binary
- lscd_compare
- lscd_graded
- wic
- wsi
```
```{tab} evaluation

- binary_wic
- change_binary
- change_graded
- compare
- none
- wic
- wsi
```
````

## Step 2

The bash will return the feedback message to ask you specify more detail infromation. For example, if you choose to use LSCD_binary model, you will be ask to specify `task/lscd_binary@task.model` with the options of `apd_compare_all` and `cos`. If you use LSCD_graded model, you will be ask to specify `task/lscd_graded@task.model` with the options of `apd_compare_all`, `apd_compare_annotated`, `cluster_jsd`, `permutation`, etc.

## Task Examples

We provide command line examples with different tasks for reference. They use the German dataset `dwug_de_210`, the German BERT and do not have evaluation setting.

### WiC

Here is the command lines for applying the WiC task. You can find more detail about WiC task in the page [Tasks/WiC](tasks/wic.md).

1. WiC task work with model `contextual_embedder` would be the following:

    ```sh
    python main.py -m \
        evaluation=none \
        task=wic \
        task/wic@task.model=contextual_embedder \
        task/wic/metric@task.model.similarity_metric=cosine \
        dataset=dwug_de_210 \
        dataset/split=dev \
        dataset/spelling_normalization=german \
        dataset/preprocessing=raw \
        task.model.ckpt=bert-base-german-cased
    ```

    <!-- 
    /mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/.venv/lib64/python3.10/site-packages/hydra/_internal/callbacks.py:26: UserWarning: Callback ResultCollector.on_multirun_end raised NotADirectoryError: [Errno 20] Not a directory: '/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/multirun/2023-05-19/17-21-55/multirun.yaml/.hydra/config.yaml' 
      warnings.warn(
    ** the directory should be '.../hh-mm-ss/0/.hydra/config.yaml'
     -->

2. WiC task work with model `deepmistake` would be the following:

    ```sh
    python main.py -m \
        evaluation=none \
        task=wic \
        task/wic@task.model=deepmistake \
        task/wic/dm_ckpt@task.model.ckpt=WIC+RSS+DWUG+XLWSD \
        dataset=dwug_de_210 \
        dataset/split=dev \
        dataset/spelling_normalization=german \
        dataset/preprocessing=raw
    ```

    <!-- 
    task/wic/dm_ckpt@task.model.ckpt=
    WIC+RSS+DWUG+XLWSD (x)
    WIC_DWUG+XLWSD (x)
    WIC_RSS (x)
    first_concat (x)
    mean_dist_l1ndotn_CE (x)
    mean_dist_l1ndotn_MSE (x)
    -->

    <!-- 
    Error in call to target 'src.wic.deepmistake.Cache':
    AttributeError("'Cache' object has no attribute 'metadata'")
    full_key: task.model.cache
    -->

### WSI

Here is the command lines for applying the WiC task. You can find more detail about WiC task in the page [Tasks/WSI](tasks/wsi.md).

```sh
python main.py -m \
    evaluation=none \
    task=wsi \
    task/wsi@task.model=cluster_correlation \
    task/wic@task.model.wic=contextual_embedder \
    task/wic/metric@task.model.wic.similarity_metric=cosine \
    dataset=dwug_de_210 \
    dataset/split=dev \
    dataset/spelling_normalization=english \
    dataset/preprocessing=raw \
    task.model.wic.ckpt=bert-base-german-cased \
    task.model.wic.gpu=1
```

<!-- 
Traceback (most recent call last):
  File "/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/src/wsi/model.py", line 35, in similarity_matrix
    similarity_matrix[i, j] = pairs_to_similarities[(use_1, use_2)]
KeyError: (Use(identifier='2532889X_1946-10-18_01_059.tcf.xml-2-15', grouping='2', context='Mit geheimnisvoller Miene flüsterten die von Michael Bohnen Entsandten jedem zu, daß heute eine Sensation zu erwarten sei.', target='Sensation', indices=(96, 105), pos='NN'), Use(identifier='2532889X_1946-10-18_01_059.tcf.xml-2-15', grouping='2', context='Mit geheimnisvoller Miene flüsterten die von Michael Bohnen Entsandten jedem zu, daß heute eine Sensation zu erwarten sei.', target='Sensation', indices=(96, 105), pos='NN'))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/main.py", line 12, in main
    return run(*instantiate(config))
  File "/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/src/utils/runner.py", line 121, in run
    predictions.update(dict(zip(ids, model.predict(uses))))
  File "/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/src/wsi/correlation_clustering.py", line 22, in predict
    similarity_matrix = self.similarity_matrix(use_pairs)
  File "/mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/src/wsi/model.py", line 37, in similarity_matrix
    similarity_matrix[i, j] = pairs_to_similarities[(use_2, use_1)]
KeyError: (Use(identifier='2532889X_1946-10-18_01_059.tcf.xml-2-15', grouping='2', context='Mit geheimnisvoller Miene flüsterten die von Michael Bohnen Entsandten jedem zu, daß heute eine Sensation zu erwarten sei.', target='Sensation', indices=(96, 105), pos='NN'), Use(identifier='2532889X_1946-10-18_01_059.tcf.xml-2-15', grouping='2', context='Mit geheimnisvoller Miene flüsterten die von Michael Bohnen Entsandten jedem zu, daß heute eine Sensation zu erwarten sei.', target='Sensation', indices=(96, 105), pos='NN'))
 -->

## LSCD

Here is the command lines for applying the WiC task. You can find more detail about WiC task in the page [Tasks/LSCD](tasks/lscd.md). There are three different version for LSCD tasks, i.e. [`lscd_binary`](1.), [`lscd_compare`](2.), and [`lscd_graded`](3.). You can find the comman line examples for each of them in the following list.

1. lscd_binary

    ```sh
    python main.py \
        dataset=dwug_de_210 \
        dataset/split=dev \
        dataset/spelling_normalization=german \
        dataset/preprocessing=raw \
        evaluation=none \
        task=lscd_binary \
        task/lscd_binary@task.model=apd_compare_all \
        task/lscd_binary/threshold_fn@task.model.threshold_fn=mean_std \
        task/wic@task.model.graded_model.wic=contextual_embedder \
        task/wic/metric@task.model.graded_model.wic.similarity_metric=cosine \
        task.model.graded_model.wic.ckpt=bert-base-german-cased \
        task.model.graded_model.wic.gpu=1
    ```

2. lscd_compare

    ```sh
    python main.py \
        dataset=dwug_de_210 \
        dataset/split=dev \
        dataset/spelling_normalization=german \
        dataset/preprocessing=raw \
        evaluation=none \
        task=lscd_compare \
        task/lscd_compare@task.model=cos \
        task/wic@task.model.wic=contextual_embedder \
        task/wic/metric@task.model.wic.similarity_metric=cosine \
        task.model.wic.ckpt=bert-base-german-cased \
        task.model.wic.gpu=1
    ```

    <!-- 
    error with task/lscd_compare@task.model=apd_compare_all
    /mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/.venv/lib64/python3.10/site-packages/hydra/_internal/callbacks.py:26: UserWarning: Callback ResultCollector.on_run_end raised NotImplementedError: 
    warnings.warn(
    Error executing job with overrides: ['dataset=dwug_de_210', 'dataset/split=dev', 'dataset/spelling_normalization=german', 'dataset/preprocessing=raw', 'evaluation=none', 'task=lscd_compare', 'task/lscd_compare@task.model=apd_compare_all', 'task/wic@task.model.wic=contextual_embedder', 'task/wic/metric@task.model.wic.similarity_metric=cosine', 'task.model.wic.ckpt=bert-base-german-cased', 'task.model.wic.gpu=1']
    Error locating target 'src.lscd.ApdCompareAll', see chained exception above.
    full_key: task.model

    * error with task/lscd_compare@task.model=cos
    /mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/.venv/lib64/python3.10/site-packages/hydra/_internal/callbacks.py:26: UserWarning: Callback ResultCollector.on_run_end raised NotImplementedError: 
    warnings.warn(
    ** solved by empty the previous output

    * error with task/lscd_compare@task.model=cluster_jsd
    /mount/arbeitsdaten20/projekte/cik/users/kuan-yu/LSCDBenchmark/.venv/lib64/python3.10/site-packages/hydra/_internal/callbacks.py:26: UserWarning: Callback ResultCollector.on_run_end raised NotImplementedError: 
    warnings.warn(
    Error executing job with overrides: ['dataset=dwug_de_210', 'dataset/split=dev', 'dataset/spelling_normalization=german', 'dataset/preprocessing=raw', 'evaluation=none', 'task=lscd_compare', 'task/lscd_compare@task.model=cluster_jsd', 'task/wsi@task.model.wsi=cluster_correlation', 'task/wic@task.model.wsi.wic=contextual_embedder', 'task/wic/metric@task.model.wsi.wic.similarity_metric=cosine', 'task.model.wsi.wic.ckpt=bert-base-german-cased', 'task.model.wsi.wic.gpu=1']
    Error in call to target 'src.lscd.cluster_jsd.ClusterJSD':
    TypeError("Can't instantiate abstract class ClusterJSD with abstract method predict_all")
    full_key: task.model
    ** after running this command lines, the warning message starts to show up
    -->

3. lscd_graded

    ```sh
    python main.py \
        dataset=dwug_de_210 \
        dataset/split=dev \
        dataset/spelling_normalization=german \
        dataset/preprocessing=raw \
        task=lscd_graded \
        task/lscd_graded@task.model=apd_compare_all \
        task/wic@task.model.wic=contextual_embedder \
        task/wic/metric@task.model.wic.similarity_metric=cosine \
        task.model.wic.ckpt=bert-base-german-cased \
        task.model.wic.gpu=1 \
        evaluation=none
    ```