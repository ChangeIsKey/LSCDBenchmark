import os
from hydra import initialize

def overrides(d: dict[str, str]) -> list[str]:
    return [f"{key}={value}" for key, value in d.items()]

def initialize_tests_hydra(version_base=None, config_path="../conf", working_dir='results'):
    # Initialize hydra config, initialization should be done only once per execution
    initialize(version_base=version_base, config_path=config_path)
    # for testing change working directory manually
    os.chdir(working_dir)
