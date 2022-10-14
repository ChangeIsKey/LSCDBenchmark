import functools
from hydra import initialize, compose
from src.utils.runner import instantiate, run as default_run


run = functools.partial(default_run, write=False)


def overrides(d: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in d.items()]


def test_graded_apd_compare_all() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(
            config_name="config",
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task/lscd_graded@task.model": "apd_compare_all",
                    "task/wic@task.model.wic": "bert",
                    "dataset": "dwug_de",
                    "dataset.test_on": "3",
                }
            ),
        )
        assert isinstance(run(*instantiate(cfg)), float)


def test_binary_apd_compare_all() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(
            config_name="config",
            overrides=overrides(
                {
                    "task": "lscd_binary",
                    "task/lscd_binary@task.model": "apd_compare_all",
                    "task/wic@task.model.graded_model.wic": "bert",
                    "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
                    "dataset": "dwug_de",
                    "dataset.test_on": "3",
                }
            ),
        )
        assert isinstance(run(*instantiate(cfg)), float)


def test_graded_cos() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(
            config_name="config",
            overrides=overrides(
                {
                    "task": "lscd_graded",
                    "task/lscd_graded@task.model": "cos",
                    "task/wic@task.model.wic": "bert",
                    "dataset": "dwug_de",
                    "dataset.test_on": "3",
                }
            ),
        )
        assert isinstance(run(*instantiate(cfg)), float)

def test_binary_cos() -> None:
    with initialize(version_base=None, config_path="../../conf"):
        cfg = compose(
            config_name="config",
            overrides=overrides(
                {
                    "task": "lscd_binary",
                    "task/lscd_binary@task.model": "cos",
                    "task/wic@task.model.graded_model.wic": "bert",
                    "task/lscd_binary/threshold_fn@task.model.threshold_fn": "mean_std",
                    "dataset": "dwug_de",
                    "dataset.test_on": "3",
                }
            ),
        )
        assert isinstance(run(*instantiate(cfg)), float)
