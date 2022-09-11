from pathlib import Path
from pandas import json_normalize
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import yaml
import json
import os

def process_experiment(experiment: Path, date_time: datetime, results: pd.DataFrame):
    try:
        score = (experiment / "score.txt").read_text()
        if score == "nan":
            return results
        score = float(score)

        config = (experiment / ".hydra" / "config.yaml").read_text()
        config = yaml.safe_load(config)
        config = json_normalize(config)

        predictions = pd.read_csv(experiment / "predictions.tsv", sep="\t", engine="pyarrow")
        n_targets = len(predictions["target"].tolist())

        row = pd.concat([config, pd.DataFrame([{"time": date_time, "score": score, "n_targets": n_targets}])], axis=1)
        if row.loc[0, "dataset.version"] == "latest":
            latest_version = sorted(datasets[row.loc[0, "dataset.name"]])[-1]
            row["dataset.version"] = latest_version
        
        return pd.concat([results, row])

    except FileNotFoundError:
        return results
        
if __name__ == "__main__":
    cwd = Path(__file__).resolve().parent
    os.chdir(cwd)

    results = pd.DataFrame()

    datasets = json.loads(Path("../datasets.json").read_text())

    outputs = Path("../outputs")
    multirun = Path("../multirun")

    for day in tqdm(list(outputs.iterdir()), desc="Processing days (outputs folder)", leave=False):
        for experiment in tqdm(list(day.iterdir()), desc="Processing experiments", leave=False):
            date_time = datetime.strptime(f"{day.name} {experiment.name}", '%Y-%m-%d %H-%M-%S')
            results = process_experiment(experiment, date_time, results)

    for day in tqdm(multirun.iterdir(), desc="Processing days (multirun folder)", leave=False):
        for time in tqdm(day.iterdir(), desc="Processing launch times", leave=False):
            for experiment in tqdm(time.iterdir(), desc="Processing experiments", leave=False):
                if experiment.is_dir():
                    date_time = datetime.strptime(f"{day.name} {time.name}", '%Y-%m-%d %H-%M-%S')
                    results = process_experiment(experiment, date_time, results) 


    first_cols = ["time", "score", "n_targets"]
    cols = first_cols  + [col for col in results.columns.tolist() if col not in first_cols]
    results = results[cols]
    results["time"] = results["time"].apply(pd.Timestamp)
    results["groupings"] = results["groupings"].astype(str)
    results["layers"] = results["layers"].astype(str)
    results["score"] = results["score"].round(decimals=10)
    results.sort_values(by="time", inplace=True, ascending=False)
    results.drop(columns=["gpu", "dataset.path"] + [col for col in results.columns if col.endswith(".module")], inplace=True)


    cols = [col for col in results.columns.tolist() if col != "time"]
    results.drop_duplicates(subset=cols, inplace=True, keep="first")

    results.to_csv("results.csv", sep="\t", index=False)



    