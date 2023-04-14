import sys
sys.path.insert(0, ".")

import functools
from hydra import initialize, compose
from src.utils.runner import instantiate, run as default_run
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


import unittest

run = functools.partial(default_run, write=False)

def overrides(d: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in d.items()]

class TestModels(unittest.TestCase):
    # @hydra.main(version_base=None, config_path="../../conf", config_name="config")
    def test_test_on(self):
        # print(DictConfig)
        #conf = OmegaConf.create()
        initialize(version_base=None, config_path="../../conf")
        cfg = compose(config_name="config", return_hydra_config=True, overrides=overrides(
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
                        "evaluation": "wic",
                        "evaluation/metric": "f1_score",
                    }
                ))
        # print(*instantiate(config))
        #HydraConfig.instance().set_config(cfg)
        #print(OmegaConf.to_yaml(cfg))
        print(cfg['hydra']['runtime'])
        return run(*instantiate(cfg))


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