"""Minimal config for HealthBench inference-only runs.

This wrapper imports healthbench datasets and can be used with:
  python run.py healthbench_infer_config.py --mode infer -w OUTPUT_DIR
"""

from mmengine.config import read_base

from opencompass.models import OpenAISDK
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

# Concurrency settings (match task3_eval.py)
num_worker = 100

# Import healthbench dataset config
with read_base():
    from opencompass.configs.datasets.HealthBench.healthbench_gen_831613 import (
        healthbench_datasets,
    )

# OC expects a top-level 'datasets' key
datasets = healthbench_datasets

# Point datasets to local copies to avoid HF download timeouts
data_dir = "/root/nathan/opencompass/data"  # adjust if your data lives elsewhere
for item in datasets:
    # original path is "huihuixu/healthbench"
    item["path"] = f"{data_dir}/huihuixu/healthbench"
    # keep threading consistent
    if "eval_cfg" in item and "evaluator" in item["eval_cfg"]:
        item["eval_cfg"]["evaluator"]["n_threads"] = num_worker

# Inference model (generates predictions)
models = [
    dict(
        abbr="gpt-oss-120b",
        type=OpenAISDK,
        path="gpt-oss-120b",  # Must match --served-model-name from vLLM
        tokenizer_path="/data1/models/gpt-oss-120b",  # Optional for vLLM API, but kept for consistency
        key="EMPTY",
        openai_api_base="http://localhost:8000/v1",  # Adjust if your vLLM is on different port
        meta_template=dict(
            round=[
                dict(role="HUMAN", api_role="HUMAN"),
                dict(role="BOT", api_role="BOT", generate=True),
            ],
        ),
        max_seq_len=16384,
        max_out_len=16384,
        batch_size=480,
        temperature=0.001,
        max_workers=num_worker,
        query_per_second=20,
        verbose=True,
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
]

# Inference runner/partitioner (enables concurrency)
infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=num_worker,
    ),
    runner=dict(
        type=LocalRunner,
        max_num_workers=num_worker,
        retry=1,
        task=dict(type=OpenICLInferTask),
    ),
)

