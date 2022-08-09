# cluster sketch

- Dataset class should be able to be removed

```yaml
gpu: null

dataset:
  name: '???'
  task: cluster
  groupings: [1, 2]
  preprocessing:
    module: src/preprocessing.py
    method: null
    params: {}
  cleaning:
    fields: {}
    method: all


model:
  name: deepmistake
  layers: [1, -1]
  layer_aggregation: average
  measure: 
    module: src/measures.py
    method: clustering_measure_1
    params:
        # for example
        n_clusters: 3
        groupings: ${dataset.groupings}
  subword_aggregation: average
```

```yaml
gpu: null

dataset:
  name: '???'
  task: lscd
  groupings: [1, 2]
  preprocessing:
    module: src/preprocessing.py
    method: null
    params: {}
  cleaning:
    fields: {}
    method: all


model:
  name: '???'
  layers: [1, -1]
  layer_aggregation: average
  measure: 
    module: src/measures.py
    method: clustering_measure_1
    params:
        # for example
        n_clusters: 3
        groupings: ${dataset.groupings}
  subword_aggregation: average
```

```py

@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: Config):
    config = Config(**OmegaConf.to_object(cfg))
    if config.model.name != "deepmistake":
      vectorizer = Vectorizer(config)
    
    dataset = DataLoader(config).load_dataset(task=config.dataset.task)

    predictions = dict()

    for target in tqdm(dataset.targets, desc="Processing targets"):
        if config.model.name != "deepmistake":
          model = VectorModel(config, vectorizer, target)
        else:
          model = DeepMistake(...)
        predictions[target.name] = config.model.measure.method(target, model, **config.model.measure.params)

    labels = (
        dataset.labels.loc[:, ["lemma", "change_graded", "change_binary"]]
        .set_index("lemma")
        .to_dict("index")
    )
    results = Results(config, predictions, labels)
    results.score(task="graded_change")


if __name__ == "__main__":
    main()






from typing import Protocol
class DistanceModel(Protocol):
    def distances(self, ids: List[Tuple[ID, ID]], method):
        ...


# clustering.py
class ClusterModel:
    """
    doesnt contain any time information
    """
    def __init__(self, use_pairs: List[Tuple[ID, ID]], distances) -> None:
        self.distances = distances  # distances come from either VectorModel or DeepMistake
        self.clusters = None

    def cluster(self, method) -> Dict[str, int]:
        """
        method: a clustering algorithm
        returns: a mapping from context identifiers to cluster labels
        """
        # 1. construct distance matrix
        # 2. cluster distance matrix
        # 3. return clusters (mapping from use ids to clusters)
        pass
    
    def split(self, groupings: Tuple[int, int], clustering: Dict[ID, int], uses_to_groupings: Dict[ID, int]) -> List[Any]:
        """
        splits clusters into two groups according to self.config.dataset.groupings
        """
        pass

    @staticmethod
    def custom_clustering_method1(...):
        ...

model = ClusterModel(distances)
model.cluster(method=sklearn.clustering.cluster_method1)


from config import pairing, sampling


# measures.py
def cluster_jsd_merge_all(target: Target, model: DistanceModel, method) -> float:
    """
    method: clustering method
    """
    pairs = sampling.all(pairing.MERGE, target)
    distances = model.distances(pairs)
    cluster_model = ClusterModel(pairs, distances)
    clusters = cluster_model.cluster(method)
    # split clusters into two sets
    # GET `uses_to_groupings` FROM TARGET
    c1, c2 = cluster_model.split(target.grouping_combination, clusters, uses_to_groupings)
    return scipy.spatial.distance.jensenshannon(c1, c2, base=2.0)



class DeepMistake:
    """
    take use pairs and maps them to distances using a classification algorithm
    without vectors
    """
    
    def __init__(self) -> None:
        pass

    def distances(self, ids: List[Tuple[ID, ID]], method):
        pass

    






```
