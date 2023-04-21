import sys
sys.path.insert(0, ".")

import functools
from hydra import initialize, compose, utils
import hydra
#from src.utils.runner import instantiate, run as default_run
from src.utils.runner import run
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy import stats

import unittest
import pytest

#run = functools.partial(default_run, write=False)

def overrides(d: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in d.items()]

class TestModels(unittest.TestCase):
    #@hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def test_apd_change_graded_german(self):
        # print(DictConfig)
        #conf = OmegaConf.create()
        initialize(version_base=None, config_path="../../conf")
        config = compose(config_name="config", return_hydra_config=True, overrides=overrides(
                    {
                        "task": "lscd_graded",
                        "task.model.wic.ckpt": "bert-base-german-cased",
                        "task/lscd_graded@task.model": "apd_compare_all",
                        "task/wic@task.model.wic": "contextual_embedder",
                        "task/wic/metric@task.model.wic.similarity_metric": "cosine",
                        #"dataset/_target_": "src.dataset.Dataset",
                        #"dataset/name": "dwug_de_210",
                        "dataset": "dwug_de_210",                        
                        "dataset/split": "dev",
                        "dataset/spelling_normalization": "german",
                        "dataset/preprocessing": "normalization",
                        # These 2 words have extreme change_graded values in the gold data: 0.0 and 0.93
                        "dataset.test_on": ["Ackergerät", "Engpaß"],
                        #"dataset/filter_lemmas": "all",
                        #"evaluation": "wic",
                        #"evaluation/metric": "f1_score",
                        "evaluation": "change_graded"

                    }
                ))
        # print(*instantiate(config))
        #HydraConfig.instance().set_config(cfg)
        #print(OmegaConf.to_yaml(cfg))
        #print(cfg['hydra']['runtime'])
        #return run(*instantiate(cfg))

        # to do: integrate this with instantiate function
        print(config.dataset)
        with open_dict(config):
            config.dataset.name = "dwug_de_210"
        
        dataset = utils.instantiate(config.dataset, _convert_="all")
        model = utils.instantiate(config.task.model, _convert_="all")
        evaluation = utils.instantiate(config.evaluation, _convert_="all")

        # to do:  add logging
        # to do:  undeerstand why Sppearman is called varioous times (plootting or permutation test?)
        # to do:  Change output directory for results

        #corr, _ = stats.spearmanr(a=[0.0, 0.9274400622952051], b=[0.22825924649553478, 0.33364990519314275], nan_policy='omit')
        #print(corr).blah
        #print(dataset, model, evaluation)
        # Run 1st time
        score1 = run(dataset, model, evaluation)
        #print(score)
        # Assert that prediction corresponds to gold
        assert pytest.approx(1.0) == score1
        # Run 2nd time
        score2 = run(dataset, model, evaluation)
        # Assert that the result reproduces acrosss runs
        assert score1 == score2


    # to do
    def test_cache(self):
        pass

                        # "dataset.test_on": "verbauen",
                        # "dataset.test_on": "vergönnen",
                        # "dataset.test_on": "voranstellen",
                        # dataset.test_on=[abbauen, abdecken, abgebrüht]

    # def test_graded_apd_compare_all(self) -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_graded",
    #                     "task/lscd_graded@task.model": "apd_compare_all",
    #                     "task/wic@task.model.wic": "bert",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         self.assertIsInstance(run(*instantiate(cfg)), float)


    # def test_binary_apd_compare_all() -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_binary",
    #                     "task/lscd_binary@task.model": "apd_compare_all",
    #                     "task/wic@task.model.graded_model.wic": "bert",
    #                     "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         assert isinstance(run(*instantiate(cfg)), float)


    # def test_graded_cos() -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_graded",
    #                     "task/lscd_graded@task.model": "cos",
    #                     "task/wic@task.model.wic": "bert",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         assert isinstance(run(*instantiate(cfg)), float)

    # def test_binary_cos() -> None:
    #     with initialize(version_base=None, config_path="../../conf"):
    #         cfg = compose(
    #             config_name="config",
    #             overrides=overrides(
    #                 {
    #                     "task": "lscd_binary",
    #                     "task/lscd_binary@task.model": "cos",
    #                     "task/wic@task.model.graded_model.wic": "bert",
    #                     "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
    #                     "dataset": "dwug_de",
    #                     "dataset.test_on": "3",
    #                 }
    #             ),
    #         )
    #         assert isinstance(run(*instantiate(cfg)), float)

if __name__ == '__main__':
    
    unittest.main()