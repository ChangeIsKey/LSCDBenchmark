import src.utils as utils
from pandas import json_normalize
from datetime import datetime
import pandas as pd
import yaml
import json
import shutil

datasets = json.loads(utils.path("datasets.json").read_text())

outputs = utils.path("outputs")
multirun = utils.path("multirun")

results = pd.DataFrame()

for day in outputs.iterdir():
    for experiment in day.iterdir():
        date_time = datetime.strptime(f"{day.name} {experiment.name}", '%Y-%m-%d %H-%M-%S')
        date = date_time.date()
        time = date_time.time()

        try:
            config = (experiment / ".hydra" / "config.yaml").read_text()
            config = yaml.safe_load(config)
            config = json_normalize(config)
            score = (experiment / "score.txt").read_text()
            if score == "nan":
                shutil.rmtree(experiment)
                continue
            score = float(score)
            row = pd.concat([config, pd.DataFrame([{"date": date, "time": time, "score": score}])], axis=1)
            if row.loc[0, "dataset.version"] == "latest":
                latest_version = sorted(datasets[row.loc[0, "dataset.name"]])[-1]
                row["dataset.version"] = latest_version
            
            row.drop(columns=["gpu"], inplace=True)

            results = pd.concat([results, row])

            
        except FileNotFoundError:
            shutil.rmtree(experiment)
            continue

first_cols = ["date", "time", "score"]
cols = first_cols  + [col for col in results.columns.tolist() if col not in first_cols]
results = results[cols]

results.to_csv("results-collected.csv", sep="\t", index=False)

        
