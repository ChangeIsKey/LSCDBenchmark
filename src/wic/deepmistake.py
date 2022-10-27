import tempfile
import json
import os
from pathlib import Path
from collections import defaultdict

from src.use import Use, to_data_format
from src.wic.model import WICModel


class DeepMistake(WICModel):
    checkpoint: Path

    def similarities(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> dict[tuple[str, str], float]:
        target2usepairs = defaultdict(list)
        for up in use_pairs:
            target2usepairs[up[0].target].append(to_data_format(up))
        target2path = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            for target, arr in target2usepairs.items():
                data = path / f"{target}.data"
                with open(data, mode="w", encoding="utf8") as f:
                    json.dump(target2usepairs[target], f)
                target2path[target] = data
            os.system()


# cd mcl-wic
# ckpt=self.checkpoint
# ckpt_path=self.checkpoint.parent

# part=tmp_dir
# python -u run_model.py --max_seq_len=500 --do_eval --ckpt_path $ckpt_path \
# --eval_input_dir $tmp_dir/ --eval_output_dir $tmp_dir/scores/$part/ \
# --output_dir $ckpt_path
