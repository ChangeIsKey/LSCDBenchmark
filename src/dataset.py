import csv
from fnmatch import translate
from gettext import translation
import json
from typing import List, Dict
from pathlib import Path
import zipfile
import requests
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import shutil

from src.config import Config
from src.lscd.target import Target
import src.utils as utils


class Dataset:

    def __init__(self, config: Config):
        self.config = config
        self.groupings = self.config.groupings

        self._uses = None
        self._judgments = None
        self._agreements = None
        self._lscd_labels = None
        self._targets = None
        self._wug_to_url = None
        self._path = None
        self.__csv_params = dict(
            delimiter="\t", encoding="utf8", quoting=csv.QUOTE_NONE
        )
        if not self.path.exists():
            if self.config.dataset.name in self.config.dataset.wug_to_url:
                self.path.parent.parent.mkdir(parents=True, exist_ok=True)
                self.__download()
                self.__unzip(self.path.parent.parent)
            else:
                raise KeyError("dataset could not be found")

    @property
    def path(self) -> Path:
        if self._path is None:
            if self.config.dataset.path is not None:
                self._path = self.config.dataset.path
            else:
                self._path = utils.path("wug") / self.config.dataset.name / self.config.dataset.version
        return self._path
            


    @property
    def translation_table(self) -> Dict[str, str]:
        dataset2lang = {
            "dwug_de": "german",
            "dwug_en": "english",
            "dwug_sv": "swedish"
        }
        
        language = dataset2lang.get(self.config.dataset.name)
        match language:
            case "german":
                translation_table = {
                    u'aͤ': u'ä', u'oͤ': u'ö', u'uͤ': u'ü', 
                    u'Aͤ': u'Ä', u'Oͤ': u'Ö', u'Uͤ': u'Ü',
                    u'ſ': u's', u'\ua75b': u'r', u'm̃': u'mm', 
                    u'æ': u'ae', u'Æ': u'Ae', u'göñ': u'gönn', 
                    u'spañ': u'spann'
                }
            # case "english":
            #     translation_table = {
            #         u' \'s': u'\'s', u' n\'t': u'n\'t', u' \'ve': u'\'ve', 
            #         u' \'d' : u'\'d', u' \'re': u'\'re', u' \'ll': u'\'ll'
            #     }
            # case "swedish":
            #     translation_table = {u' \'s': u'\'s'}
            case _:
                translation_table = {}
        return translation_table 

    @property
    def url(self) -> str:
        return self.config.dataset.wug_to_url[self.config.dataset.name][self.config.dataset.version]
        
    @property
    def __zipped_filename(self) -> Path:
        return utils.path(f"{self.config.dataset.name}-{self.config.dataset.version}.zip")
    
    def __download(self) -> None:
        r = requests.get(self.url, stream=True)
        with open(file=self.__zipped_filename, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading dataset '{self.config.dataset.name}' (version {self.config.dataset.version})",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(r.headers["Content-Length"]),
                leave=False
            )
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()

    def __unzip(self, output_dir: Path) -> None:
        zipped = self.__zipped_filename
        with zipfile.ZipFile(file=zipped) as z:
            namelist = z.namelist()
            root = output_dir / namelist[0]
            root.mkdir(parents=True, exist_ok=True)

            for filename in tqdm(namelist, desc=f"Unzipping dataset '{self.config.dataset.name}' (version {self.config.dataset.version})"):
                path = Path(filename)
                f_parts = list(path.parts)
                f_parts[f_parts.index(root.name)] = self.config.dataset.version
                target_path = root.joinpath(*f_parts)
                

                if not filename.endswith("/"):
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with target_path.open(mode="wb") as file_obj:
                        shutil.copyfileobj(z.open(filename, mode="r"), file_obj)
                    
        zipped.unlink()

    @property
    def lscd_labels(self) -> DataFrame:
        if self._lscd_labels is None:
            stats_groupings =  "stats_groupings.csv"
            path = self.path / "stats" / "semeval" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / "opt" / stats_groupings
            if not path.exists():
                path = self.path / "stats" / stats_groupings
            self._lscd_labels = pd.read_csv(path, delimiter="\t", encoding="utf8")
        return self._lscd_labels

    @property
    def agreements(self) -> DataFrame:
        if self._agreements is None:
            path = self.path / "stats" / "stats_agreement.csv"
            self._agreements = pd.read_csv(path, **self.__csv_params)
        return self._agreements
    
    @agreements.setter
    def agreements(self, other: DataFrame) -> None:
        self._agreements = other
    
    @property
    def uses(self):
        if self._uses is None:
            self._uses = pd.concat([target.uses for target in self.targets])
        return self._uses

    @property
    def judgments(self):
        if self._judgments is None:
            self._judgments = pd.concat([target.judgments for target in self.targets])
        return self._judgments

    @property
    def clusters(self):
        if self._clusters is None:
            self._clusters = pd.concat([target.clusters for target in self.targets])
        return self._clusters

    @property
    def targets(self) -> List[Target]:
        if self._targets is None:
            to_load = []

            if self.config.test_targets is not None:
                to_load = self.config.test_targets
            elif len(self.config.cleaning.stats) > 0:
                self.agreements = self.agreements.iloc[1:, :].copy()  # remove "data=full" row
                self.agreements = self.config.cleaning(self.agreements)
                to_load = self.agreements.data.unique().tolist()
            else:
                to_load = [f.name for f in (self.path / "data").iterdir()]

            trans_table = self.translation_table
            self._targets = [
                Target(config=self.config, name=target, translation_table=trans_table, path=self.path)
                for target in tqdm(to_load, desc="Building targets", leave=False)
            ]

        return self._targets
