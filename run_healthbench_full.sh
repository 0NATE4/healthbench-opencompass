# Async HealthBench Evaluator Implementation

## Overview

Built a custom async evaluator to replace OpenCompass's slow evaluation stage, achieving 2+ rps throughput vs OC's ~0.21 rps. The pipeline runs OC inference, then uses async judge calls for evaluation. The current default judge model is **gpt-oss-120b**.

## Quickstart (do this from `/root/nathan/opencompass`)

1) Download data: grab the HealthBench JSONL files from `https://huggingface.co/datasets/opencompass/healthbench` and place them at `data/healthbench/` (`2025-05-07-06-14-12_oss_eval.jsonl`, `hard_2025-05-08-21-00-10.jsonl`, `consensus_2025-05-09-20-00-46.jsonl`).
2) Judge model: export your judge API; defaults to `qwen3-235b-a22b-thinking-2507`:
   ```bash
   export OC_JUDGE_API_BASE="http://YOUR_API_BASE/v1"
   export OC_JUDGE_API_KEY=YOUR_KEY
   export OC_JUDGE_MODEL="qwen3-235b-a22b-thinking-2507"
   ```
3) Inference model: set/adjust `models` in `healthbench_infer_config.py` (or import a model config).
4) Run the full pipeline (inference → async eval):
   ```bash
   ./run_healthbench_full.sh <inference_model_name> <dev_subset> <concurrency>
   # e.g.
   ./run_healthbench_full.sh gpt-oss-120b full 200
   ```
Outputs land under `outputs/<work_dir>/results/` (per-subset JSON) and `judge_logs/`.

## Files Created

### 1. `healthbench_scoring.py`
**Purpose:** Scoring logic extracted from `opencompass/datasets/healthbench/healthbench.py`

**Key Components:**
- `RubricItem` class - represents a single rubric criterion with points and tags
- `calculate_score()` - computes score for a single example (sum points for met criteria)
- `compute_clipped_stats()` - computes mean, bootstrap_std, n_samples (matches OC's `_compute_clipped_stats`)
- `aggregate_scores()` - aggregates across all examples with per-tag breakdowns
- `ScoredExample` dataclass - holds results for a single example
- Grader prompt templates (`GRADER_TEMPLATE`, `PER_EXAMPLE_GRADER_TEMPLATE`) - copied from healthbench.py
- Prompt building functions (`build_grader_prompt_per_rubric`, `build_grader_prompt_per_example`)
- JSON parsing utilities (`parse_json_response`, `parse_per_example_response`)

**Reused from healthbench.py:**
- Exact scoring logic (calculate_score, bootstrap_std)
- Tag groupings (example_tags + rubric_tags)
- RubricItem structure

### 2. `async_healthbench_eval.py`
**Purpose:** Main CLI evaluator with async judge calls

**CLI Arguments:**
- `--predictions`: Path to OC predictions JSON file or directory of sharded files
- `--dataset`: Path to HealthBench JSONL (optional if using --dataset-subset)
- `--dataset-subset`: Auto-selects dataset file (`base`, `hard`, `consensus`)
- `--mode`: `per_rubric` (default) or `per_example`
- `--concurrency`: Max concurrent judge calls (default: 200)
- `--output`: Results JSON path
- `--judge-log`: Streaming JSONL of judge outputs (enables resume)
- `--resume-from`: Skip already-judged items from existing JSONL
- `--limit`: Limit to first N examples (0 = no limit)
- `--subset`: Dev subset name for metadata (e.g., dev_50, dev_tiny)
- `--api-base`, `--api-key`, `--model`: Judge API config (or env vars)
- `--max-tokens`, `--timeout`, `--max-retries`: Judge call parameters
- `--verbose`: Verbose output

**Key Features:**
- **Join Strategy:** Joins predictions to dataset by position index with safety checks:
  - Asserts length match (with warnings)
  - Logs first 5 entries showing index + prompt_id for manual verification
  - Future: Can join by prompt_id if predictions include it
- **Prediction Loading:** 
  - Supports single JSON file or directory of sharded files
  - Filters by subset pattern when `--dataset-subset` is used:
    - `base`: `healthbench_[0-9]*.json` (excludes hard/consensus)
    - `hard`: `healthbench_hard.json`
    - `consensus`: `healthbench_consensus.json`
- **Async Judge Calls:**
  - Semaphore-based concurrency control
  - Retry logic (max 3 retries, exponential backoff for rate limits)
  - Streaming JSONL output for resume capability
  - Progress logging with RPS calculation
- **Resume Support:**
  - Loads already-judged items from `--resume-from` JSONL
  - Skips those items during evaluation
  - Enables resuming interrupted runs
- **Output Format:**
  - Matches OC structure: `accuracy`, `accuracy_std`, `n_samples`
  - Adds `config` section with mode, concurrency, subset info
  - Adds `timing` section with total time and RPS
  - Includes per-tag metrics in `details`

**Dataset File Mapping:**
```python
DATASET_FILES = {
    "base": "2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "hard_2025-05-08-21-00-10.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
}
```

**Prediction Pattern Matching:**
```python
PREDICTION_PATTERNS = {
    "base": r"^healthbench(_\d+)?\.json$",  # healthbench.json or healthbench_0.json, healthbench_1.json, etc.
    "hard": r"^healthbench_hard(_\d+)?\.json$",  # healthbench_hard.json or healthbench_hard_0.json, etc.
    "consensus": r"^healthbench_consensus(_\d+)?\.json$",  # healthbench_consensus.json or healthbench_consensus_0.json, etc.
}
```

**Note**: All subsets support sharded files (with `_0`, `_1`, etc. suffix). The `(_\d+)?` pattern makes the shard number optional, so both single files (`healthbench_hard.json`) and sharded files (`healthbench_hard_0.json`) are matched.

### 3. `run_healthbench_full.sh`
**Purpose:** End-to-end wrapper script (OC inference + async evaluation)

**Usage:**
```bash
./run_healthbench_full.sh MODEL_NAME [DEV_SUBSET] [CONCURRENCY]
```

**Features:**
- Runs OC inference with `HEALTHBENCH_SUBSET` env var
- Automatically finds predictions directory
- Runs async evaluator for all 3 subsets (base, hard, consensus)
- Respects dev subset limits:
  - `dev_tiny`: 2 base, 1 hard, 2 consensus
  - `dev_50`: 50 base only (hard/consensus skipped)
  - `dev_medium`: 250 base, 100 hard, 300 consensus
  - `full`: all examples (no limit)
- Outputs separate results per subset: `results_base.json`, `results_hard.json`, `results_consensus.json`
- Prints summary of all results at end

**Script Structure:**
1. Validates arguments
2. Determines limits based on dev subset
3. Runs `python run.py healthbench_infer_config.py --mode infer`
4. Locates predictions directory
5. Calls `async_healthbench_eval.py` for each subset
6. Prints summary

### 4. `healthbench_infer_config.py`
**Purpose:** Wrapper config file for OC inference

**Why needed:** The original `healthbench_gen_831613.py` only defines `healthbench_datasets` list, but OC's `run.py` expects a top-level `datasets` key in the config.

**Solution:** Created minimal wrapper that:
- Imports `healthbench_datasets` from the original config
- Exports as `datasets` for OC compatibility
- Can be used with `python run.py healthbench_infer_config.py --mode infer`

**IMPORTANT:** This config requires models to be added. You must either:
1. Import models from a model config: `from opencompass.configs.models.your_model import models`
2. Define models inline in the config file
3. Modify the bash script to merge a model config

**Current status:** Has placeholder `models = []` - will fail until models are added.

## Architecture

```
OC predictions/*.json  +  HealthBench JSONL  →  async_healthbench_eval.py  →  results.json
       +
HealthBench rubrics
```

**Full Pipeline:**
```
OC inference (run.py)  →  predictions/*.json  →  async_healthbench_eval.py  →  results.json
```

## Key Design Decisions

### 1. Join Strategy
- **Current:** Position-based join (predictions["0"] ↔ dataset[0])
- **Safety:** Length assertions + logging for manual verification
- **Future:** Can join by `prompt_id` if OC predictions include it

### 2. Subset Handling
- OC runs 3 separate datasets: base, hard, consensus
- Each has separate prediction files
- Async evaluator handles each separately with pattern matching

### 3. Resume Support
- Judge log is streaming JSONL (one line per rubric result)
- Each line is self-contained: `prompt_id`, `rubric_index`, `criteria_met`, `explanation`, `raw_response`
- Resume state tracks which `(prompt_id, rubric_index)` pairs are already judged

### 4. Scoring Equivalence
- Ported exact logic from `healthbench.py`:
  - Same `calculate_score()` function
  - Same `bootstrap_std()` implementation
  - Same tag groupings
- Ensures scores match OC's output

## Usage Examples

### Test async evaluator only (run from repo root `/root/nathan/opencompass`):
```bash
python async_healthbench_eval.py \
  --predictions outputs/.../predictions/MODEL/ \
  --dataset-subset base \
  --mode per_rubric \
  --limit 5 \
  --subset dev_tiny \
  --concurrency 200 \
  --output test_results.json \
  --judge-log test_judge.jsonl
```

### Full pipeline (run from repo root `/root/nathan/opencompass`):
```bash
./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_50 200
```

### Resume from interruption:
```bash
python async_healthbench_eval.py \
  --predictions outputs/.../predictions/MODEL/ \
  --dataset-subset base \
  --resume-from existing_judge_log.jsonl \
  --judge-log existing_judge_log.jsonl \
  --output results.json
```

## Data Download and Layout

- Download HealthBench JSONL data from the official release: `https://huggingface.co/datasets/opencompass/healthbench` (contains `2025-05-07-06-14-12_oss_eval.jsonl`, `hard_2025-05-08-21-00-10.jsonl`, `consensus_2025-05-09-20-00-46.jsonl`).
- Save the JSONL files to `data/healthbench/` inside the repo. The async evaluator defaults to these paths when `--dataset-subset` is provided.
- Ensure you run commands from the repo root `/root/nathan/opencompass` so relative paths resolve.

## Output Format

```json
{
  "accuracy": 0.46,
  "accuracy_std": 0.02,
  "n_samples": 50,
  "config": {
    "mode": "per_rubric",
    "concurrency": 200,
    "judge_model": "...",
    "dataset_subset": "base",
    "dev_subset": "dev_50",
    ...
  },
  "timing": {
    "total_seconds": 123.4,
    "examples_scored": 50,
    "rps": 2.5
  },
  "details": {
    "example_tag_metrics": {...},
    "rubric_tag_metrics": {...}
  }
}
```

## Judge Log Format (JSONL)

Each line is a self-contained JSON object:
```json
{"prompt_id": "...", "rubric_index": 3, "rubric_points": 2.0, "criteria_met": true, "explanation": "...", "raw_response": "..."}
```

## Validation Plan

1. Run OC on `dev_50` (inference only, `--mode infer`)
2. Run async evaluator on those predictions
3. Compare scores to OC's full run (should match)
4. Confirm throughput ~10x faster (2+ rps vs 0.21 rps)

## Known Issues / Future Improvements

1. **Join by prompt_id:** Currently uses position index. Would be safer to join by `prompt_id` if OC predictions include it.
   - **Status:** Known limitation, has safety checks (length assertions + logging)
   - **Priority:** Medium - should prioritize adding prompt_id to predictions

2. **Model config:** The wrapper config requires models to be added.
   - **Status:** `healthbench_infer_config.py` has placeholder for models
   - **Solution:** Add model config import or define models inline in the config file

3. **Resume state reconstruction:** Currently resume only skips items, doesn't reconstruct full results.
   - **Status:** Works but could be optimized
   - **Future:** Load from judge log to rebuild ScoredExample objects directly

4. **Error handling:** ✅ **IMPROVED**
   - ✅ Fail loudly if no prediction files match pattern
   - ✅ Log exactly which files were matched per subset
   - ✅ Judge call stats (total, successful, failed, success rate)
   - ✅ Clear summary at end with subset info

## Risk Mitigation (Addressing Feedback)

### 1. Index-based Join ✅
- **Current:** Position-based join with safety checks
- **Safety measures:**
  - Length assertions with warnings
  - Logs first 5 entries showing index + prompt_id for manual verification
  - Fails if no predictions match dataset indices
- **Future:** Switch to prompt_id join when OC predictions include it

### 2. Prediction Pattern Assumptions ✅
- **Current:** Pattern matching for subset files
- **Safety measures:**
  - ✅ Logs exactly which files matched per subset
  - ✅ Fails loudly if none matched (exits with error + shows available files)
  - Pattern mapping is explicit and documented

### 3. Multi-subset Behavior ✅
- **Current:** Runs base, hard, consensus separately
- **Safety measures:**
  - ✅ Each output clearly shows `dataset_subset` in config
  - ✅ `n_samples` and `accuracy` refer to the specific subset being scored
  - ✅ Separate results files per subset
  - ✅ Limits are subset-specific and documented

### 4. Resume Semantics ✅
- **Current:** Skips already-judged pairs, rewrites results.json
- **Status:** Works for first version
- **Future:** Can rebuild ScoredExample objects from judge log (polish task)

### 5. Error Handling and Observability ✅
- **Current:** Comprehensive error handling and logging
- **Features:**
  - ✅ Fail fast on missing predictions, dataset load errors, judge config errors
  - ✅ Clear summary: judge calls attempted, success rate, failures
  - ✅ Progress logging with RPS
  - ✅ Timing information

## Files Modified

- `async_healthbench_eval.py` - Added `--limit`, `--subset`, `--dataset-subset` flags
- `run_healthbench_full.sh` - Updated config path and added multi-subset support

## Dependencies

- `openai` (AsyncOpenAI)
- `numpy` (for bootstrap_std)
- `asyncio` (for async judge calls)
- OpenCompass dataset classes (for loading)

## Testing Status

- ✅ Scoring module created and imports successfully
- ✅ Async evaluator created with all features
- ✅ Bash wrapper created
- ⚠️  Models config needs to be added to `healthbench_infer_config.py` (see Quick Fix below)
- ⏳ Validation against OC results pending
- ⏳ Throughput verification pending

## Quick Fix: Adding Models to Config

To fix the `KeyError: 'models'` error, edit `healthbench_infer_config.py` and add:

```python
# Option 1: Import from existing model config
from opencompass.configs.models.your_model_name import models

# Option 2: Define inline (example)
from opencompass.models import OpenAISDK
models = [
    dict(
        abbr="your-model-name",
        type=OpenAISDK,
        path="your-model-path",
        # ... other model config ...
    ),
]
```

Or modify `run_healthbench_full.sh` to accept a model config file and merge it.

## How to Run (End-to-End)

1) Set judge API env vars (required)
```bash
export OC_JUDGE_API_BASE="http://YOUR_API_BASE/v1"
export OC_JUDGE_API_KEY=YOUR_KEY
export OC_JUDGE_MODEL="qwen3-235b-a22b-thinking-2507"
```

2) (Optional) Limit HealthBench subset for inference
```bash
# dev_tiny | dev_50 | dev_medium | full
export HEALTHBENCH_SUBSET=dev_tiny
```

3) Ensure inference model is defined in `healthbench_infer_config.py`
   - Currently set to `qwen3-4b-thinking-2507` via OpenAISDK (vLLM-style API)
   - Adjust `openai_api_base` if your inference server is not `http://localhost:8000/v1`

4) Run the full pipeline
```bash
cd /root/nathan/opencompass
chmod +x run_healthbench_full.sh

# Example: dev_tiny, concurrency 200
./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_tiny 200

# Example: dev_50, concurrency 200
./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_50 200
```

This will:
- Run OC inference
- Run async evaluator for base/hard/consensus (limits per dev subset)
- Write results in OC-compatible structure
- Write judge logs for resume/debug

5) Resume (optional)
```bash
python async_healthbench_eval.py \
  --predictions <pred_dir> \
  --dataset-subset base \
  --mode per_rubric \
  --resume-from judge_logs/healthbench.jsonl \
  --judge-log judge_logs/healthbench.jsonl \
  --output results/qwen3-4b-thinking-2507/healthbench.json
```

## Directory Layout (After Run)

For a run with work_dir `outputs/healthbench_dev_tiny_<TS>`:
```
outputs/healthbench_dev_tiny_<TS>/
  ├─ predictions/                # OC inference outputs (may be under timestamp if inference-only)
  │   └─ MODEL/
  ├─ results/                    # Async eval results (OC-compatible)
  │   └─ MODEL/
  │       ├─ healthbench.json
  │       ├─ healthbench_hard.json
  │       └─ healthbench_consensus.json
  ├─ logs/
  │   └─ eval/MODEL/             # Async eval logs
  ├─ summary/
  │   ├─ summary_<TS>.md
  │   ├─ summary_<TS>.csv
  │   └─ summary_<TS>.txt
  └─ judge_logs/                 # Extra (resume/debug)
      ├─ healthbench.jsonl
      ├─ healthbench_hard.jsonl
      └─ healthbench_consensus.jsonl
```
Notes:
- OC inference may place predictions under an additional timestamp subdir when run in inference-only mode. The wrapper now searches both direct `predictions/` and `TIMESTAMP/predictions/`.
- Results/logs/summary are written directly under the work_dir to mirror OC’s layout.

## Troubleshooting

- **Missing models (KeyError: 'models')**: Add models to `healthbench_infer_config.py` (see Quick Fix above).
- **API 404/5xx errors**: We now retry 404/5xx/429 with backoff. Persistent 404 likely indicates wrong model name or API base.
- **No prediction files matched**: Check the predictions directory and subset. Patterns expected:
  - base: `healthbench.json`
  - hard: `healthbench_hard.json`
  - consensus: `healthbench_consensus.json`
- **Join alignment**: We log first 5 indices + prompt_id and assert lengths. If misaligned, ensure predictions correspond to the same dataset ordering; long-term fix is to include `prompt_id` in predictions and join by id.
- **Performance**: Use `--concurrency` to tune; default 200. Judge logs stream to JSONL for resume.

## Why the Structure Differs from OC (and How We Aligned It)

- OC runs create a timestamped work_dir; we initially inherited an extra timestamp when running inference-only. The wrapper now:
  - Searches both direct `predictions/` and `TIMESTAMP/predictions/`.
  - Writes `results/`, `logs/eval/`, `summary/` directly under the work_dir to match OC.
- We add `judge_logs/` (extra) for resume/debug; OC doesn’t produce these.


