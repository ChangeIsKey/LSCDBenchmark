WARNING: Outdated documentation

# LSCDBenchmark

## Creating an environment
This tool relies on [graph-tool](https://graph-tool.skewed.de/) for some of its functionality. This library is notoriously difficult to install via regular methods.
However, Docker provides a simple way to create a development environment. See the Wiki for more details.



## Details on dataset:

- COMPARE score for datasets only exists for datasets where word-usage pairs for annotation were randomly sampled (e.g., durel, dwug_es)
- stats_groupings.tsv always contains statistics for group comparison. Groups represent in most cases time periods (e.g., dwug_de, dwug_en), but in some cases other distinctions, such as dialect, can be made (e.g., diawug).
- Most datasets contain only two groups (e.g., two time periods), but some contain more groups (e.g., diawug, dups-wug)

## Walkthrough

This benchmark relies heavily on [Hydra](hydra.cc) to allow flexibility in the
experiments. This means that any parameter specified in the config can be
changed via the command line (and some of them have to be provided mandatorily,
since they have no default).
For more information on how to modify configuration parameters, refer to the
Hydra [docs](https://hydra.cc/docs/advanced/override_grammar/basic/).  The default configuration can be found in
[config/defaults.yaml](config/defaults.yaml). We don't recommend modifying this
file. Instead, modify its parameters through the CLI.

---

The only required parameters in the shell are the dataset name (`dataset.name`)
and the model identifier (`model.name`). This identifier should be the identifier
of a model from the [Huggingface Hub](https://huggingface.co/models).

A basic command could therefore be the following:

`python main.py dataset.name=dwug_de model.name=bert-base-german-cased`

Which would result in the following final configuration for a given experiment:

```yaml
dataset:
  name: dwug_de
  task: lscd
  groupings: [1, 2]
  preprocessing:
    module: src/preprocessing.py
    method: null
    params: {}
  cleaning:
    stats: {}
    method: all


model:
  gpu: null
  name: bert-base-german-uncased
  layers: [1, -1]
  layer_aggregation: average
  measure: 
    module: src/measures.py
    method: apd_compare_all
    params: {}
  subword_aggregation: average


hydra:
  job:
    chdir: true
```


### Preprocessing

The default configuration performs no preprocessing on the contexts (`dataset.preprocessing.method: null`). To change this behaviour, you can either specify a method implemented in the [preprocessing module](src/preprocessing.py), such as `toklem`, `lemmatize` and `tokenize`, or implement your own preprocessing module, specify its path, and specify the name of the new method. If your method requires extra parameters, you can provide them in the `preprocessing.params` dictionary.

### Cleaning

The default configuration performs no cleaning (filtering) of the target words. It is possible to filter some targets out by specifying some fields. For example, if we wanted to filter out targets whose Krippendorf's alpha is below 0.3, we can do:

> `python main.py dataset.name=dwug_de model.name=bert-base-german-cased +dataset.cleaning.stats.kri_full.threshold=0.3`

This would result in the following configuration:

```yaml
dataset:
  name: dwug_de
  task: lscd
  groupings: [1, 2]
  preprocessing:
    module: src/preprocessing.py
    method: null
    params: {}
  cleaning:
    stats:
      kri_full:
        threshold: 0.3
        keep: above
    method: all


model:
  gpu: null
  name: bert-base-german-uncased
  layers: [1, -1]
  layer_aggregation: average
  measure: 
    module: src/measures.py
    method: apd_compare_all
    params: {}
  subword_aggregation: average


hydra:
  job:
    chdir: true
```

Notice the `keep` field under `kri_full`. If it's not provided, this field will
be filled automatically. It determines whether we should keep the targets above
the specified threshold (alternative, you can set it to `below`).
These `stats` can be any numerical column in the `stats_agreement.tsv` files, and
you can add as many as necessary. The `cleaning.method` parameter specifies
whether to keep targets that fulfill all conditions, or to keep those that
fulfill at least one.

### Layers

You can specify which model layers to use as embeddings for the targets, and how to aggregate them. Currently, you can average them (`average`), concatenate them (`concat`), and add them together (`sum`).

### Subwords

You can also specify how to aggregate the embeddings for the subwords corresponding to the target word. You can either average them (`average`), take the first (`first`) or the last (`last`), or add them together (`sum`).

### Measures

You can specify on which measure to evaluate the targets. As with the preprocessing, you can create your own module, specify its path and the name of the new method (plus any other extra parameters), or use one of the methods provided in the [measures module](src/measures.py).


## Things to keep in mind

There are some options that will eventually go into the config:

- The evaluation metrics (e.g., spearman, f1) might work in a similar fashion as with the preprocessing or the measures
- ...
