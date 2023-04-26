# LSCDBenchmark Documentation

Welcome to the LSCDBenchmark documentation. This home page contains an index with a brief description of the different sections in the documentation.

Lexical Semantic Change Detection (LSCD) is a field of NLP that studies methods automating the analysis of changes in word meanings over time. In recent years, this field has seen much development in terms of models, datasets and tasks [[1]](#1). This has made it hard to keep a good overview of the field. Additionally, with the multitude of possible options for preprocessing, data cleaning, dataset versions, model parameter choice or tuning, clustering algorithms, and change measures a shared testbed with common evaluation setup is needed in order to precisely reproduce experimental results. Hence, we present a benchmark repository implementing evaluation procedures for models on most available LSCD datasets.

```{toctree}
:maxdepth: 1
:caption: CONTENTS:

getting-started
datasets
tasks/index
leaderboard
usage
```

## Indices and tables

- {ref}`genindex`
- [Glossary](glossary.md)
- {ref}`search`

## References

<a id="1">[1]</a>
Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart.
