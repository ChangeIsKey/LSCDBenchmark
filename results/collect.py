from pathlib import Path
from typing import Any
from pandas import json_normalize
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import yaml
import json
import os

def collect_results() -> pd.DataFrame:
    results = pd.DataFrame()

    datasets = json.loads(Path("../datasets.json").read_text())

    outputs = Path("../outputs")
    multirun = Path("../multirun")

    for day in tqdm(list(outputs.iterdir()), desc="Processing days (outputs folder)", leave=False):
        for experiment in tqdm(list(day.iterdir()), desc="Processing experiments", leave=False):
            date_time = datetime.strptime(f"{day.name} {experiment.name}", '%Y-%m-%d %H-%M-%S')
            results = process_experiment(experiment, date_time, results, datasets)

    for day in tqdm(multirun.iterdir(), desc="Processing days (multirun folder)", leave=False):
        for time in tqdm(day.iterdir(), desc="Processing launch times", leave=False):
            for experiment in tqdm(time.iterdir(), desc="Processing experiments", leave=False):
                if experiment.is_dir():
                    date_time = datetime.strptime(f"{day.name} {time.name}", '%Y-%m-%d %H-%M-%S')
                    results = process_experiment(experiment, date_time, results, datasets) 


    # reorder columns
    results["score"] = results["score"].round(decimals=10)
    first_cols = ["time", "score", "n_targets"]
    cols = first_cols  + [col for col in results.columns.tolist() if col not in first_cols]
    results = results[cols]
    return results


def process_experiment(experiment: Path, date_time: datetime, results: pd.DataFrame, datasets: dict[str, Any]):
    try:
        score = (experiment / "score.txt").read_text()
        if score == "nan":
            return results
        score = float(score)

        config = (experiment / ".hydra" / "config.yaml").read_text()
        config = yaml.safe_load(config)


        if config["evaluation"]["task"] != "change_binary":
            config["evaluation"]["binary_threshold"]["module"] = None
            config["evaluation"]["binary_threshold"]["method"] = None
            config["evaluation"]["binary_threshold"]["params"] = None
        
        if len(config["cleaning"]["stats"]) == 0:
            config["cleaning"]["method"] = None

        if config["measure"]["method"] != "cluster_jsd":
            config["measure"]["clustering"]["module"] = None
            config["measure"]["clustering"]["method"] = None
            config["measure"]["clustering"]["params"] = None
        
        if config["dataset"]["version"] == "latest":
            latest_version = sorted(datasets[config["dataset"]["name"]])[-1]
            config["dataset"]["version"] = latest_version
            
        predictions = pd.read_csv(experiment / "predictions.tsv", sep="\t", engine="pyarrow")
        n_targets = len(predictions["target"].tolist())

        row = pd.concat([
            json_normalize(config), 
            pd.DataFrame([{"time": date_time, "score": score, "n_targets": n_targets}])
        ], axis=1)
        return pd.concat([results, row])

    except FileNotFoundError:
        return results
        
if __name__ == "__main__":
    cwd = Path(__file__).resolve().parent
    os.chdir(cwd)

    results = collect_results()
    results.to_csv("results.csv", sep="\t", index=False)
    results.to_excel("results.xlsx", index=False)



    