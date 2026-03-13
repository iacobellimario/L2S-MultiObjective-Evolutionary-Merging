import os
import json
import sys
import pandas as pd
import numpy as np
from logging import getLogger
from typing import List, Optional, Dict
from transformers import AutoTokenizer
import subprocess

logger = getLogger(__name__)

QWEN_EVAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../Qwen2.5-Math/evaluation")
)

class QwenEvaluator:
    def __init__(
        self,
        datasets: List[str],
        max_samples: int = 20,
        lang_id: Optional[str] = None,
        device: str = "cuda",
        output_root: Optional[str] = None,
        seed: int = 42,
        split: str = "train"
    ):
        self.datasets = datasets
        self.max_samples = max_samples
        self.lang_id = lang_id
        self.device = device
        self.seed = int(seed)
        self.data = pd.DataFrame()
        self.split = split
        self.output_root = output_root or os.path.join(os.getcwd(), "qwen_eval_outputs")
        os.makedirs(self.output_root, exist_ok=True)

    def _run_math_eval(self, model_path: str, dataset: str) -> str:
        print("MODEL_PATH:", model_path)        

        # When multiple datasets are passed (comma-separated), the evaluator is
        # expected to write one subdirectory per dataset under `output_root`.
        out_dir = os.path.join(self.output_root, f"{dataset}")
       
        cmd = [
            sys.executable,
            "math_eval.py",
            "--model_name_or_path", model_path,
            "--data_names", dataset,
            "--prompt_type", "deepseek-math",
            "--max_tokens_per_call", str(10240),
            "--num_test_sample", str(self.max_samples),
            "--n_sampling", "1",
            "--split", self.split,
            "--seed", str(self.seed),
            "--output_dir", str(self.output_root),
            "--save_outputs",
            "--overwrite",
            "--use_safetensors",
            "--use_vllm"
        ]        
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=QWEN_EVAL_DIR, check=True)
        return out_dir
        

    def _collect_outputs(self, out_dir: str):
        """
        Walk an output directory and parse any `.jsonl` files.
        """
        records = []
        for fname in sorted(os.listdir(out_dir)):
            path = os.path.join(out_dir, fname)
            if fname.endswith(".jsonl"):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            records.append(json.loads(line))
                        except Exception:
                            # Skip malformed lines but keep parsing the rest.
                            continue
        return records

    def evaluate(self, model_path: str) -> pd.DataFrame:
        datasets_str = ",".join(self.datasets)
        self._run_math_eval(model_path, datasets_str)
                
        all_dfs = []
        for ds in self.datasets:
            ds_out_dir = os.path.join(self.output_root, ds)
            recs = self._collect_outputs(ds_out_dir)
            rows = []
            if recs:
                for r in recs:
                    correctness = r.get("score")[0] if isinstance(r.get("score"), list) else r.get("score")
                    code = r.get("code")
                    model_ans = " ".join(code) if isinstance(code, list) else (str(code) if code is not None else "")

                    rows.append({
                        "correctness": correctness,
                        "model_answers": model_ans,
                    })
            df_ds = pd.DataFrame(rows)
          
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                def safe_token_len(x):
                    if x is None or (isinstance(x, str) and x.strip() == ""):
                        return 0
                    return len(tokenizer.encode(str(x), add_special_tokens=False))
                df_ds["answer_length"] = df_ds["model_answers"].apply(safe_token_len)
            except Exception as e:
                logger.warning(f"Tokenizer unavailable or failed: {e}. Falling back to whitespace split.")
                def safe_word_len(x):
                    if x is None or (isinstance(x, str) and x.strip() == ""):
                        return 0
                    return len(str(x).split())
                df_ds["answer_length"] = df_ds["model_answers"].apply(safe_word_len)
           
            all_dfs.append(df_ds)
           
        if all_dfs:
            self.data = pd.concat(all_dfs, ignore_index=True)
            return self.data
        else:
            self.data = pd.DataFrame()
            return self.data

    def get_data(self) -> pd.DataFrame:
        if hasattr(self, "data") and isinstance(self.data, pd.DataFrame):
            return self.data
        else:
            return pd.DataFrame()
