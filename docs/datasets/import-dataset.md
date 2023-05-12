# Import A New Dataset

You can import a new data set to apply for LSCDBenchmark.

## Build A .yaml file

Firstly, you need to inform the benchmark about the detail of the data set you want to import. Build a `dataset_name.yaml` and fill it in the information of the data set.

Here are the content structure of `.yaml`. Only `path:`, `standard_split:`, `type:`, and `url:` need to be filled with the information of the data set you want to import. To put either `???` or `none` in `spelling_normalization:` under `defaults:` is optional. You may keep all other exsisting information in the following example.

```yaml
_target_: src.dataset.Dataset 
cleaning: null
defaults:
- preprocessing: ???
- spelling_normalization: ??? # or 'none' (do not include single quotes)
- split: ???
exclude_annotators: []
groupings:
- "1"
- "2"
wic_use_pairs:
group: ALL
sample: annotated
path: testwug_en # path to save the data set
standard_split: # fill with words in the data set
dev:
    - afternoon_nn
    - arm
    - plane_nn
    - target
dev1: []
dev2: []
full:
    - afternoon_nn
    - arm
    - plane_nn
    - target
test: []
test_on: null
type: dev # depend on the standard_split
url: https://zenodo.org/record/7900959/files/testwug_en.zip # the url to download the dataset
```

Take importing *testwug_en* data set as an example. You can firstly define `path:` and `url:`. We have `path: testwug_en`, so that the data set will be download through the url and saved in the `./wug/testwug_en` directory. All the data set will saved in `./wug/` by defult.

Then, regarding to filling `standard_split:`, you can either split your words in the data set into *develop set* or *test set*. All the words need to be included in `full:`. If you want to split the word in *develop set*, you can put them in `dev:`. You can even further spilt them into two groups: `dev1:` and `dev2:`. You can only either fill these two groups with the words in `dev:` or stay with empty list.

Lastly, according to how you want to split your data set, fill in either `dev` or `test` in `type:`. In our case, we split our data set in *develop set*, so we have `type: dev`.

After you create the `dataset_name.yaml`, you may see your data set option in the list after you excute the folloing commend[^1]:

```sh
python main.py\
    evaluation=none\
    task=wic\
    task/wic@task.model=contextual_embedder\
    task/wic/metric@task.model.similarity_metric=dot
```

[^1]: All the setting in the commend above is choosed randomly in order to display the data set option list. It is not a complete commend example to excute benchmark.
