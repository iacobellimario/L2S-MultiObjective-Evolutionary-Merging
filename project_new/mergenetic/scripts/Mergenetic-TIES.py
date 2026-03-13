# ...existing code...
#!/usr/bin/env python3

# ==== Imports ====
# pymoo components
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from mergenetic.merging.ties_merger import TiesMerger
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.duplicate import DefaultDuplicateElimination

# Mergekit and Mergenetic
import mergenetic
from mergenetic.searcher import Searcher
from mergenetic.utils import ConfigLmEval, Config
from mergenetic.optimization.predefined_problems import L2SMathReasoningProblem, ConfigQwen
from mergenetic import PROJECT_ROOT

# lm_eval
from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer

# Hugging Face 
#from huggingface_hub import whoami
#from huggingface_hub import notebook_login
#from huggingface_hub import snapshot_download

import os, random, numpy as np, torch
import sys, importlib, pathlib

# ========== Setup environment ==========
real_pkg_dir = pathlib.Path(f"{PROJECT_ROOT}/project_new/mergenetic/src/mergenetic")
if not real_pkg_dir.exists():
    raise RuntimeError("mergenetic/src/mergenetic not found – is the repo cloned?")

# Clean previously loaded modules
for name in list(sys.modules):
    if name == "mergenetic" or name.startswith("mergenetic."):
        del sys.modules[name]

# Setup sys.path
src_root = str(real_pkg_dir.parent)  
if src_root not in sys.path:
    sys.path.insert(0, src_root)

for bad in ("", "/content"):
    if bad in sys.path:
        sys.path.remove(bad)

importlib.invalidate_caches()
import mergenetic
print("✅ Loaded mergenetic from:", getattr(mergenetic, '__file__', 'N/A'))

# ========== Set SEED for reproducuility =====
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Environment variables
os.environ["PYTHONHASHSEED"] = str(SEED)
print(f"All seeds set to {SEED}")

# ========== Model setup ==========
model_dir = "/leonardo_scratch/fast/IscrC_LENS/miacobel/model"
os.makedirs(model_dir, exist_ok=True)

# ========== Config ==========
config = Config()
config.bench = None
config.seed = SEED
config.run_id = "Mergenetic-TIES-7B-MATH-Entropy"
config.device = "cuda" 
config.task_type = "FG_MATH"
config.path_to_store_config = f"{PROJECT_ROOT}/project_new/mergenetic/experiments/evolutionary-merging-qwen"
config.path_to_store_merged_model = f"{model_dir}/merged"
config.base_model = f"{model_dir}/DeepSeek-R1-Distill-Qwen-7B"
config.models = {"en": f"{model_dir}/Qwen2.5-Math-7B"}
config.mode = "mean"
config.eval_batch_size = 8

# ========== Qwen datasets sampling configuration ==========
QWEN_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "project_new", "mergenetic", "qwen_outputs", config.run_id)
os.makedirs(QWEN_OUTPUT_ROOT, exist_ok=True)

# ========== Estimation Parameters ==========
est_parameters = ConfigQwen(
    qwen_datasets = ["math"],
    qwen_split = "entropy-test-7B",
    qwen_max_samples = 50,
    qwen_seed = SEED,
    qwen_output_root = QWEN_OUTPUT_ROOT,
)
print(f"✅ Qwen evaluation configured: datasets={est_parameters.qwen_datasets}, K={est_parameters.qwen_max_samples}, output_root={est_parameters.qwen_output_root}")

# ========== Merger ==========
path_to_store_yaml = f"{config.path_to_store_config}/{config.run_id}"
lang_id = "en"

merger = TiesMerger(
    run_id=config.run_id,
    path_to_base_model=config.base_model,
    model_paths=[config.models[lang_id]],
    path_to_store_yaml=path_to_store_yaml,
    path_to_store_merged_model=config.path_to_store_merged_model,
    dtype=config.dtype,    
)

# ========== Problem ==========
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True) # tokenizer for length computation

problem = L2SMathReasoningProblem(
    merger=merger,
    test_df=None,
    search_df=None,
    lang_id=lang_id,
    conf_pe=est_parameters,
    device=config.device,
    n_var=2, # (w,d) for TIES
    n_obj=2, # (-accuracy, length) 
    n_eq_constr=0,
    n_ieq_constr=0,
    discrete=True,
    eval_batch_size=config.eval_batch_size,
    lm_eval_tasks=None,
    tokenizer=tokenizer
)

# ========== Search ==========
config.pop_size = 20
config.n_iter = 10
run_id = config.run_id


algorithm = NSGA2(
    pop_size=config.pop_size,
    sampling=IntegerRandomSampling(),
    crossover=SBX(),
    mutation=PM(),
    eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-2)
)
"""
algorithm = GA(
    pop_size=config.pop_size,
    sampling=IntegerRandomSampling(),
    crossover=SBX(),
    mutation=PM(),
    eliminate_duplicates=DefaultDuplicateElimination(epsilon=1e-2)
)
"""
results_path = f"{config.path_to_store_config}/{run_id}/"
os.makedirs(results_path, exist_ok=True)

searcher = Searcher(
    problem=problem,
    n_iter=config.n_iter,
    algorithm=algorithm,
    results_path=results_path,
    run_id=config.run_id,
    seed=config.seed,
    verbose=False,
)

result_df = searcher.search()
print("✅ Evolutionary search completed. Results in:", results_path)
