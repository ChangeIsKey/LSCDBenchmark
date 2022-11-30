import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import _TypedDict, Any, TypedDict
from git import Repo

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from pydantic import BaseModel, HttpUrl, PrivateAttr
from tqdm import tqdm
import subprocess

from src.use import Use, UseID
from src.utils import utils
from src.wic.model import WICModel
from logging import getLogger

log = getLogger(__name__)


class Model(BaseModel):
    name: str
    url: HttpUrl


class Cache(BaseModel):
    metadata: dict[str, Any]
    _similarities: DataFrame = PrivateAttr(default=None)
    _similarities_filtered: DataFrame = PrivateAttr(default=None)
    __metadata__: DataFrame = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.__metadata__ = pd.json_normalize(self.metadata)

        try:
            self._similarities = pd.read_csv(
                filepath_or_buffer=self.path, engine="pyarrow"
            )
            self._similarities.drop_duplicates(inplace=True)
        except FileNotFoundError:
            self._similarities = self.__metadata__.assign(
                use_0=None, use_1=None, similarity=None, lemma=None
            )
            self._similarities = self._similarities.iloc[0:0]  # remove first dummy row
        finally:
            self._similarities["similarity"] = self._similarities["similarity"].astype(
                float
            )
            self._similarities["use_0"] = self._similarities["use_0"].astype(str)
            self._similarities["use_1"] = self._similarities["use_1"].astype(str)
            self._similarities["lemma"] = self._similarities["lemma"].astype("category")
        
        self._similarities_filtered = self._similarities.merge(self.__metadata__)
        

    def retrieve(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> dict[tuple[UseID, UseID], float]:
        lookup_table = DataFrame(
            {
                "use_0": [up[0].identifier for up in use_pairs],
                "use_1": [up[1].identifier for up in use_pairs],
                "lemma": [up[1].target for up in use_pairs],
            }
        )
        merged = lookup_table.merge(self._similarities_filtered, how="inner")
        return dict(zip(list(zip(merged["use_0"], merged["use_1"])), merged["similarity"]))

    def persist(self) -> None:
        self._similarities = pd.concat([self._similarities, self._similarities_filtered], ignore_index=True)
        self._similarities.drop_duplicates(inplace=True)
        self._similarities.to_csv(self.path, index=False)
        

    @property
    def path(self) -> Path:
        cache = os.getenv("DEEPMISTAKE")
        if cache is None:
            cache = ".deepmistake"
        return utils.path(cache) / "similarities.csv"

    def add_use_pair(self, use_pair: tuple[Use, Use], similarity: float) -> None:
        row = self.__metadata__.assign(
            use_0=use_pair[0].identifier,
            use_1=use_pair[1].identifier,
            similarity=similarity,
            lemma=use_pair[0].target,
        )
        self._similarities_filtered = pd.concat([self._similarities_filtered, row], ignore_index=True)


class Score(TypedDict):
    id: str
    score: tuple[str, str] | str


def use_pair_group(use_pair: tuple[Use, Use]) -> str:
    if use_pair[0].grouping != use_pair[1].grouping:
        return "COMPARE"
    else:
        if use_pair[0].grouping == 0 and use_pair[1].grouping == 0:
            return "EARLIER"
        else:
            return "LATER"


class Input(TypedDict):
    id: str
    start1: int
    end1: int
    sentence1: str
    start2: int
    end2: int
    sentence2: str
    lemma: str
    pos: str
    grp: str


def to_data_format(use_pair: tuple[Use, Use]) -> Input:
    return {
        "id": f"{use_pair[0].target}.{np.random.randint(low=100000, high=1000000)}",
        "start1": use_pair[0].indices[0],
        "end1": use_pair[0].indices[1],
        "sentence1": use_pair[0].context,
        "start2": use_pair[1].indices[0],
        "end2": use_pair[1].indices[1],
        "sentence2": use_pair[1].context,
        "lemma": use_pair[0].target,
        "pos": "NOUN" if use_pair[0].pos == "NN" else use_pair[0].pos,
        "grp": use_pair_group(use_pair),
    }


class DeepMistake(WICModel):
    ckpt: Model
    cache: Cache | None

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache is not None:
            self.cache.persist()

    def clone_repo(self) -> None:
        Repo.clone_from(url="https://github.com/ameta13/mcl-wic", to_path=self.repo_dir)

    @property
    def path(self) -> Path:
        path = os.getenv("DEEPMISTAKE")
        if path is None:
            path = ".deepmistake"
        return utils.path(path)
    
    @property
    def repo_dir(self) -> Path:
        return self.path / "mcl-wic"

    @property
    def ckpt_dir(self) -> Path:
        return self.path / "checkpoints" / self.ckpt.name

    def __unzip_ckpt(self, zipped: Path) -> None:
        with zipfile.ZipFile(file=zipped) as z:
            namelist = z.namelist()[1:]  # remove root element

            for filename in tqdm(
                namelist, desc="Unzipping checkpoint files", leave=False
            ):
                filename_p = Path(filename)
                path = self.ckpt_dir / filename_p.parts[-1]
                with path.open(mode="wb") as file_obj:
                    shutil.copyfileobj(z.open(filename, mode="r"), file_obj)

        zipped.unlink()

    def __download_ckpt(self) -> Path:
        filename = self.ckpt.url.split("/")[-1]
        assert filename.endswith(".zip")

        r = requests.get(self.ckpt.url, stream=True)
        ckpt_dir = self.ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / filename

        with open(file=path, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading checkpoint '{self.ckpt.name}'",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(r.headers["Content-Length"]),
                leave=False,
            )
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()
        return path

    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        with self:
            if not self.ckpt_dir.exists():
                zipped = self.__download_ckpt()
                self.__unzip_ckpt(zipped)

            data_dir = self.ckpt_dir / "data"
            output_dir = self.ckpt_dir / "scores"

            output_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)

            scores: dict[tuple[UseID, UseID], float | None] = {
                (use_pair[0].identifier, use_pair[1].identifier): None
                for use_pair in use_pairs
            }

            non_cached_use_pairs: list[tuple[Use, Use]] = []

            inputs = [to_data_format(use_pair) for use_pair in use_pairs]
            # list of pairs of use ids to index, deepmistake-formatted input, similarity, and original data
            data: dict[tuple[UseID, UseID], tuple[Input, tuple[Use, Use]],] = {
                (use_pair[0].identifier, use_pair[1].identifier): (
                    inputs[i],
                    use_pair,
                )
                for i, use_pair in enumerate(use_pairs)
            }

            input_id_to_use_pair_ids = {
                inputs[i]["id"]: (use_pair[0].identifier, use_pair[1].identifier)
                for i, use_pair in enumerate(use_pairs)
            }

            if self.cache is not None:
                scores.update(self.cache.retrieve(use_pairs))
                for pair, similarity in scores.items():
                    if similarity is None:
                        non_cached_use_pairs.append(data[pair][1])

            if len(non_cached_use_pairs) > 0:
                hydra_dir = os.getcwd()
                os.chdir(self.ckpt_dir)

                input_ = [
                    data[(up[0].identifier, up[1].identifier)][0]
                    for up in non_cached_use_pairs
                ]

                path = data_dir / f"{use_pairs[0][0].target}.data"
                with open(path, mode="w", encoding="utf8") as f:
                    json.dump(input_, f)

                if not self.repo_dir.exists():
                    self.clone_repo()

                script = self.repo_dir / "run_model.py"

                # run run_model.py and capture output (don't print it)
                subprocess.check_output(
                    f"python -u {script} \
                    --max_seq_len=500 \
                    --do_eval \
                    --ckpt_path {self.ckpt_dir} \
                    --eval_input_dir {data_dir} \
                    --eval_output_dir {output_dir} \
                    --output_dir {output_dir}", 
                    shell=True, 
                    # if the script doesn't run, comment out the next line
                    stderr=subprocess.PIPE
                )

                path.unlink()

                with open(
                    file=output_dir / f"{use_pairs[0][0].target}.scores",
                    encoding="utf8",
                ) as f:
                    dumped_scores: list[Score] = json.load(f)
                    for x in dumped_scores:
                        id_ = x["id"]
                        if len(x["score"]) == 2:
                            score_0 = float(x["score"][0])
                            score_1 = float(x["score"][1])
                            similarity = np.mean([score_0, score_1]).item()
                        else:
                            similarity = float(x["score"][0])

                        use_pair = input_id_to_use_pair_ids[id_]
                        scores[use_pair] = similarity
                        if self.cache is not None:
                            self.cache.add_use_pair(
                                use_pair=data[use_pair][1], similarity=similarity
                            )

                os.chdir(hydra_dir)

            results = [scores[(up[0].identifier, up[1].identifier)] for up in use_pairs] 
            if any([score is None for score in results]):
                if self.cache is not None:
                    self.cache.persist()
                return self.predict(use_pairs)
            self.predictions = scores
            return results