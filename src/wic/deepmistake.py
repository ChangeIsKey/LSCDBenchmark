import json
import os
import numpy as np
from pathlib import Path

from src.use import Use, to_data_format
from src.wic.model import WICModel
from src.utils import utils


class DeepMistake(WICModel):
    ckpt: Path

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.ckpt = utils.path(str(self.ckpt))

    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        data_dir = self.ckpt.parent / "data"
        output_dir = self.ckpt.parent / "scores"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

        data = [to_data_format(up) for up in use_pairs]
        path = data_dir / f"{use_pairs[0][0].target}.data"
        with open(path, mode="w", encoding="utf8") as f:
            json.dump(data, f)

        script = utils.path("src") / "wic" / "mcl-wic" / "run_model.py"

        hydra_dir = os.getcwd()

        os.chdir(self.ckpt.parent)
        os.system(
            f"python -u {script} \
            --max_seq_len=500 \
            --do_eval \
            --ckpt_path {self.ckpt.parent} \
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
