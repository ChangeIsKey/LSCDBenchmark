import sys
sys.path.insert(0, ".")

import os
from hydra import initialize, compose, utils
from tests.utils import overrides
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest

class TestWICModels(unittest.TestCase):

    def test_wic_ger_ackergeraet_engpass(self) -> None:
        
        # Compose hydra config
        initialize(version_base=None, config_path="../../conf")
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        "task/wic/dm_ckpt@task.model.ckpt": "WIC_DWUG+XLWSD",
                        "task/wic@task.model": "deepmistake",
                        "dataset": "dwug_de_210",                        
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "german",
                        "dataset/preprocessing": "normalization",
                        # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.93
                        "dataset.test_on": ["Ackergerät", "Engpaß"],
                        "evaluation": "wic",
                        "evaluation/metric": "spearman",
                        "evaluation/plotter": "none",
                    }
                ))

        # Run 1st time
        score1 = run(*instantiate(config))
        # Assert that prediction corresponds to gold
        print(score1)
        assert pytest.approx(1.0) == score1
        # Run 2nd time
        score2 = run(*instantiate(config))
        # Assert that the result reproduces across runs
        assert score1 == score2


if __name__ == '__main__':
    
    # Initialize hydra config, initialization should be done only once per execution
    initialize(version_base=None, config_path="../../conf")
    # for testing change working directory manually
    os.chdir('results')
    unittest.main()
