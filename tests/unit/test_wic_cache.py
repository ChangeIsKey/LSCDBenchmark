import sys
sys.path.insert(0, ".")

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
from src.wic.model import WICModel

from hydra import initialize, compose, utils
from tests.utils import overrides, initialize_tests_hydra
from src.utils.runner import instantiate, run
from scipy import stats

import unittest
import pytest


class TestPredictorCache(unittest.TestCase):
    def setUp(self):
        initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results')

        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "wic",
                        # "task/wic/dm_ckpt@task.model.ckpt": "WIC_DWUG+XLWSD",
                        "task/wic@task.model": "contextual_embedder",
                        "task.model.ckpt": "pierluigic/xl-lexeme",
                        # "task.model.ckpt": "bert-base-cased",
                        "dataset": "dwug_en_200",
                        # "dataset": "testwug_en_111",
                        "task/wic/metric@task.model.similarity_metric": "cosine",                 
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "english",
                        "dataset/preprocessing": "raw",
                        # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.93
                        # "dataset.test_on": ["Ackergerät", "Engpaß"],
                        # "dataset.test_on": ["arm"],
                        "evaluation": "wic",
                        "evaluation/metric": "spearman",
                        # "evaluation/plotter": "none",
                    }
                ))

        _, self.predictor, _ = instantiate(config)
        self.predictor._cache = pd.DataFrame({
            "use_0": ["a", "b"],
            "use_1": ["x", "y"],
            "prediction": [0.5, 0.6],
        })
        self.query = pd.DataFrame({
            "use_0": ["a", "b", "c"],
            "use_1": ["x", "y", "z"],
            "prediction": [0.5, 0.6, None],
        })

    def test_merge_cache(self):
        expected_result = pd.DataFrame({
            "use_0": ["a", "b", "c"],
            "use_1": ["x", "y", "z"],
            "prediction": [0.5, 0.6, None],
        })

        result = self.predictor._merge_cache(self.query)
        
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_non_cached_no_duplicates(self):
        df = pd.DataFrame({
            "use_0": ["a", "b", "c"],
            "use_1": ["x", "y", "z"],
            "prediction": [0.5, 0.6, None]
        })
        expected_result = pd.DataFrame({
            "use_0": ["c"],
            "use_1": ["z"],
            "prediction": [None]
        }).reset_index(drop=True)
        expected_result['prediction'] = expected_result['prediction'].astype('float64')
        
        result = self.predictor._get_non_cached(df)
        result['prediction'] = result['prediction'].astype('float64')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_non_cached_with_duplicates(self):
        df = pd.DataFrame({
            "use_0": ["a", "b", "c", "c"],
            "use_1": ["x", "y", "z", "z"],
            "prediction": [0.5, 0.6, None, None],
        })
        expected_result = pd.DataFrame({
            "use_0": ["c", "c"],
            "use_1": ["z", "z"],
            "prediction": [None, None],
        }).reset_index(drop=True)
        expected_result['prediction'] = expected_result['prediction'].astype('float64')
        
        result = self.predictor._get_non_cached(df)
        result['prediction'] = result['prediction'].astype('float64')
        pd.testing.assert_frame_equal(result, expected_result)


    def test_update_cache(self):
        non_cached = pd.DataFrame({
            "use_0": ["c"],
            "use_1": ["z"],
            "prediction": [None],
        })
        new_predictions = [0.7]
        self.predictor._update_cache(non_cached, new_predictions)
        self.predictor._cache['prediction'] = self.predictor._cache['prediction'].astype('float64')
        expected_cache = pd.DataFrame({
            "use_0": ["a", "b", "c"],
            "use_1": ["x", "y", "z"],
            "prediction": [0.5, 0.6, 0.7]
        })

        pd.testing.assert_frame_equal(self.predictor._cache, expected_cache)

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_parquet")
    def test_save_cache(self, mock_to_parquet, mock_mkdir):
        cache_path = Path("test_cache.parquet")
        self.predictor._cache_path = cache_path

        self.predictor._save_cache()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        mock_to_parquet.assert_called_once_with(cache_path, index=False)

if __name__ == '__main__':
    unittest.main()
