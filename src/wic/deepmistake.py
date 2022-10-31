import tempfile
import json
import os
from pathlib import Path
from collections import defaultdict

from src.use import Use, to_data_format
from src.wic.model import WICModel


class DeepMistake(WICModel):
    checkpoint: Path

    def predict(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> list[float]:
        data = [to_data_format(up) for up in use_pairs]
        dir = Path("deepmistake")
        output = dir / "scores"
        path = dir / f"{use_pairs[0][0].target}.data"

        with open(path, mode="w", encoding="utf8") as f:
            json.dump(data, f)

        scores_path = dir / "scores"
        os.system(f"python -u run_model.py \
            --max_seq_len=500 \
            --do_eval \
            --ckpt_path {self.checkpoint.parent} \
            --eval_input_dir {dir} \
            --eval_output_dir {output} \
            --output_dir {output}") 
            
        return [] 


# cd mcl-wic
# ckpt=self.checkpoint
# ckpt_path=self.checkpoint.parent

# part=tmp_dir
# python -u run_model.py --max_seq_len=500 --do_eval --ckpt_path $ckpt_path \
# --eval_input_dir $tmp_dir/ --eval_output_dir $tmp_dir/scores/$part/ \
# --output_dir $ckpt_path
