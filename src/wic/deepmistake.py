import json
import os
from pathlib import Path
import shutil
from typing import Literal, TypedDict

import zipfile
import numpy as np
import requests
from pydantic import BaseModel, HttpUrl
from tqdm import tqdm

from src.use import Use, to_data_format
from src.utils import utils
from src.wic.model import WICModel

class Model(BaseModel):
    name: str
    url: HttpUrl

class DeepMistake(WICModel):
    ckpt: Model

    @property
    def cache_dir(self) -> Path:
        cache = os.getenv("DEEPMISTAKE_CKPTS")
        if cache is None:
            cache = ".deepmistake"
        return utils.path(cache)
        
    @property
    def ckpt_dir(self) -> Path:
        return self.cache_dir / self.ckpt.name

    def __unzip_ckpt(self, zipped: Path) -> None:
        with zipfile.ZipFile(file=zipped) as z:
            namelist = z.namelist()[1:]  # remove root element

            for filename in tqdm(namelist, desc="Unzipping checkpoint files", leave=False):
                filename_p = Path(filename)
                path = self.ckpt_dir / filename_p.parts[-1]
                with path.open(mode="wb") as file_obj:
                    shutil.copyfileobj(z.open(filename, mode="r"), file_obj)

        zipped.unlink()


    def __download_ckpt(self) -> Path:
        filename = self.ckpt.url.split('/')[-1]
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
        if not self.ckpt_dir.exists():
            zipped = self.__download_ckpt()
            self.__unzip_ckpt(zipped)

        data_dir = self.ckpt_dir / "data"
        output_dir = self.ckpt_dir / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        data = [to_data_format(up) for up in use_pairs]
        path = data_dir / f"{use_pairs[0][0].target}.data"
        with open(path, mode="w", encoding="utf8") as f:
            json.dump(data, f)

        script = utils.path("src") / "wic" / "mcl-wic" / "run_model.py"

        hydra_dir = os.getcwd()

        os.chdir(self.ckpt_dir)
        os.system(
            f"python -u {script} \
            --max_seq_len=500 \
            --do_eval \
            --ckpt_path {self.ckpt_dir} \
            --eval_input_dir {data_dir} \
            --eval_output_dir {output_dir} \
            --output_dir {output_dir}"
        )
        path.unlink()

        with open(
            file=output_dir / f"{use_pairs[0][0].target}.scores", encoding="utf8"
        ) as f:
            dumped_scores: list[dict[str, str | list[str]]] = json.load(f)
            scores = []
            for x in dumped_scores:
                score_0 = float(x["score"][0])
                score_1 = float(x["score"][1])
                scores.append(np.mean([score_0, score_1]).item())

        os.chdir(hydra_dir)

        return scores
