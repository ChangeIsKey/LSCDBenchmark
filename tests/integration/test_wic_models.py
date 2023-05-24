import sys
sys.path.insert(0, ".")

from hydra import initialize, compose, utils
from tests.utils import overrides
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest

class TestModels(unittest.TestCase):
    
    def some_test_function(self) -> None:

    python main.py -m \
    dataset=dwug_de_210 \
    dataset/preprocessing=toklem \
    dataset/spelling_normalization=german \
    dataset/split=dev \
    'dataset.test_on=[abbauen,abdecken,"abgebrüht"]' \
    task=wic \
    evaluation=wic \
    evaluation/metric=spearman \
    task/wic@task.model=deepmistake \
    task/wic/dm_ckpt@task.model.ckpt=WIC_DWUG+XLWSD

    ,mean_dist_l1ndotn_CE,mean_dist_l1ndotn_MSE,WIC_DWUG+XLWSD,WIC_RSS,WIC+RSS+DWUG+XLWSD


        # Initialize and compose hydra config
        initialize(version_base=None, config_path="../../conf")
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "lscd_graded",
                        "task.model.wic.ckpt": "bert-base-german-cased",
                        "task/lscd_graded@task.model": "apd_compare_all",
                        "task/wic@task.model.wic": "contextual_embedder",
                        "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                        "dataset": "dwug_de_210",                        
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "german",
                        "dataset/preprocessing": "normalization",
                        "dataset.test_on": ["Ackergerät", "Engpaß"],
                        "evaluation": "change_graded"
                        "evaluation/plotter": "none",
                    }
                ))

        # Run 
        score1 = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score1

    def some_other_test_function(self) -> None:
        pass

if __name__ == '__main__':
    
    unittest.main()
