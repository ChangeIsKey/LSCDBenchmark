import numpy as np
from scipy.spatial import distance
from functools import singledispatchmethod
from tqdm import tqdm

from src.lscd.lscd_model import LSCDModel
from src.target import Target
from src.use import Use


class Cos(LSCDModel):
    @singledispatchmethod
    def predict(self, target):
        raise NotImplementedError
    
    @predict.register(list)
    def predict_many(self, targets: list[Target]) -> float:
        predictions = []
        for target in tqdm(targets):
            predictions.append(self.predict_one(target))
        return predictions
    
    @predict.register(Target)
    def predict_one(self, target: Target) -> float:
        earlier = target.uses[target.uses.grouping == target.grouping_combination[0]].apply(Use.from_series, axis=1).tolist()
        later = target.uses[target.uses.grouping == target.grouping_combination[1]].apply(Use.from_series, axis=1).tolist()

        earlier_vectors = np.vstack([self.wic_model.encode(use) for use in earlier])
        later_vectors = np.vstack([self.wic_model.encode(use) for use in later])

        earlier_avg = earlier_vectors.mean(axis=0)
        later_avg = later_vectors.mean(axis=0)

        return -distance.cosine(earlier_avg, later_avg)

