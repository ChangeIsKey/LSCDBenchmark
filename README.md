# LSCDBenchmark

## Details on dataset:
- COMPARE score for datasets only exists for datasets where word-usage pairs for annotation were randomly sampled (e.g., durel, dwug_es)
- stats_groupings.tsv always contains statistics for group comparison. Groups represent in most cases time periods (e.g., dwug_de, dwug_en), 
but in some cases other distinctions, such as dialect, can be made (e.g., diawug).
- Most datasets contain only two groups (e.g., two time periods), but some contain more groups (e.g., diawug, dups-wug)

## Quick start
So far two baselines have been implemented: BERT and XLM-R. Regarding the datasets, all datasets except for dwug_la should work with all implemented features.
To run a baseline, run ``