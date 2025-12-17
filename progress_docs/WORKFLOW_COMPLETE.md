# Complete HealthBench Evaluation Workflow

## Overview

This document provides a complete, step-by-step guide for evaluating an LLM on HealthBench using:
1. **OpenCompass** for inference (generating model predictions)
2. **Async HealthBench Evaluator** for evaluation (judge scoring)

**Goal**: Replace OpenCompass's slow evaluation (4 days) with fast async evaluation while maintaining identical results.

---

## Prerequisites

### 1. OpenCompass Installation

```bash
# Clone and install OpenCompass (if not already done)
cd /root/nathan/opencompass
pip install -e .
```

### 2. HealthBench Dataset Files

Ensure dataset files are in the correct location:

```bash
data/huihuixu/healthbench/
├── 2025-05-07-06-14-12_oss_eval.jsonl      # Base subset
├── hard_2025-05-08-21-00-10.jsonl          # Hard subset
└── consensus_2025-05-09-20-00-46.jsonl     # Consensus subset
```

**Location**: Set in `healthbench_infer_config.py` line 28:
```python
data_dir = "/root/nathan/opencompass/data"  # adjust if your data lives elsewhere
```

### 3. Inference Server (vLLM)

Your inference model must be running as a vLLM server:

```bash
# Example: vLLM server should be running on
http://localhost:8000/v1
```

**Model Configuration**: Edit `healthbench_infer_config.py` lines 37-59:
- `abbr`: Model name (appears in output paths)
- `path`: Model identifier
- `tokenizer_path`: Path to tokenizer
- `openai_api_base`: vLLM server URL (default: `http://localhost:8000/v1`)

### 4. Judge API Access

Judge model API must be accessible:

```bash
# Set these environment variables:
export OC_JUDGE_API_BASE="http://YOUR_API_BASE/v1"
export OC_JUDGE_API_KEY="your_api_key_here"
export OC_JUDGE_MODEL="qwen3-235b-a22b-thinking-2507"
```

---

## Complete Workflow

### Step 1: Configure Inference Model

**File**: `healthbench_infer_config.py`

**What to edit**:
- Lines 37-59: Model configuration (abbr, path, tokenizer_path, openai_api_base)
- Line 28: Dataset directory path (if different from default)

**Example**:
```python
models = [
    dict(
        abbr="your-model-name",           # Change this
        type=OpenAISDK,
        path="your-model-name",
        tokenizer_path="/path/to/tokenizer",
        openai_api_base="http://localhost:8000/v1",  # Your vLLM server
        # ... rest of config
    ),
]
```

### Step 2: Set Judge API Environment Variables

```bash
export OC_JUDGE_API_BASE="http://YOUR_API_BASE/v1"
export OC_JUDGE_API_KEY="your_api_key"
export OC_JUDGE_MODEL="qwen3-235b-a22b-thinking-2507"
```

**Verify**:
```bash
echo $OC_JUDGE_API_BASE
echo $OC_JUDGE_MODEL
```

### Step 3: Choose Dataset Subset (Optional)

**Options**:
- `dev_tiny`: 2 base, 1 hard, 2 consensus (for quick testing)
- `dev_50`: 50 base only (for benchmarking)
- `dev_100`: 50 base, 25 hard, 25 consensus (~100 total)
- `dev_medium`: 250 base, 100 hard, 300 consensus
- `full`: All examples (no limit) - **This is the full dataset**

**Default**: If not specified, uses `dev_50`

### Step 4: Run Full Pipeline

**Command**:
```bash
cd /root/nathan/opencompass
chmod +x run_healthbench_full.sh

./run_healthbench_full.sh MODEL_NAME [DEV_SUBSET] [CONCURRENCY]
```

**Arguments**:
- `MODEL_NAME`: Model abbreviation (must match `abbr` in `healthbench_infer_config.py`)
- `DEV_SUBSET`: Optional. One of: `dev_tiny`, `dev_50`, `dev_100`, `dev_medium`, `full` (default: `dev_50`)
- `CONCURRENCY`: Optional. Max concurrent judge calls (default: 200)

**Examples**:

```bash
# Quick test (dev_tiny)
./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_tiny 200

# Benchmark (dev_50)
./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_50 200

# Full dataset evaluation
./run_healthbench_full.sh qwen3-4b-thinking-2507 full 200

# Higher concurrency (faster, more API load)
./run_healthbench_full.sh qwen3-4b-thinking-2507 full 300
```

### Step 5: What Happens During Execution

**Phase 1: OpenCompass Inference**
1. Script sets `HEALTHBENCH_SUBSET` env var (if not `full`)
2. Runs: `python run.py healthbench_infer_config.py --mode infer -w OUTPUT_DIR`
3. OpenCompass generates predictions for all three subsets (base, hard, consensus)
4. Predictions saved to: `OUTPUT_DIR/TIMESTAMP/predictions/MODEL_NAME/`
5. Files created: `healthbench_0.json`, `healthbench_1.json`, ... (sharded)

**Phase 2: Async Evaluation**
1. Script locates predictions directory automatically
2. For each subset (base, hard, consensus):
   - Loads predictions (handles sharded files with reindexing)
   - Loads corresponding dataset JSONL
   - Runs async judge calls (per-rubric mode)
   - Saves results to: `OUTPUT_DIR/results/MODEL_NAME/`
3. Generates summary files (markdown, CSV, text)
4. Saves timing information to `OUTPUT_DIR/timing.txt`

**Output Structure**:
```
outputs/healthbench_full_20251209_145131/
├── 20251209_145138/                    # OC inference timestamp
│   ├── predictions/
│   │   └── qwen3-4b-thinking-2507/
│   │       ├── healthbench_0.json
│   │       ├── healthbench_1.json
│   │       ├── ...
│   │       ├── healthbench_hard_0.json
│   │       ├── ...
│   │       ├── healthbench_consensus_0.json
│   │       └── ...
│   └── logs/infer/
│       └── qwen3-4b-thinking-2507/
│           ├── healthbench_0.out
│           ├── healthbench_1.out
│           └── ...
├── results/                            # Async eval results
│   └── qwen3-4b-thinking-2507/
│       ├── healthbench.json            # Base subset results
│       ├── healthbench_hard.json      # Hard subset results
│       └── healthbench_consensus.json # Consensus subset results
├── logs/eval/                          # Async eval logs
│   └── qwen3-4b-thinking-2507/
│       ├── healthbench.out
│       ├── healthbench_hard.out
│       └── healthbench_consensus.out
├── summary/                            # Summary files
│   ├── summary_healthbench_full_20251209_145131.md
│   ├── summary_healthbench_full_20251209_145131.csv
│   └── summary_healthbench_full_20251209_145131.txt
├── judge_logs/                         # Judge outputs (for resume)
│   ├── healthbench.jsonl
│   ├── healthbench_hard.jsonl
│   └── healthbench_consensus.jsonl
└── timing.txt                          # Pipeline timing
```

### Step 6: Understanding Results

**Result Files** (`results/MODEL_NAME/*.json`):

Each result file contains:
```json
{
  "accuracy": 0.5157,
  "accuracy_std": 0.0456,
  "n_samples": 50,
  "details": {
    "example_tag_metrics": {...},    # Per-tag scores
    "rubric_tag_metrics": {...}      # Per-rubric-tag scores
  },
  "config": {
    "mode": "per_rubric",
    "concurrency": 200,
    "judge_model": "...",
    "dataset_subset": "base",
    "dev_subset": "dev_100",
    ...
  },
  "timing": {
    "total_seconds": 301.9,
    "examples_scored": 50,
    "rps": 0.17
  },
  "judge_stats": {
    "total_calls": 539,
    "successful": 539,
    "failed": 0,
    "success_rate": 1.0
  }
}
```

**Timing File** (`timing.txt`):
```
HealthBench Pipeline Timing
============================
Model: qwen3-4b-thinking-2507
Dev Subset: full
Concurrency: 200

Inference Time:  Xh Ym Zs (total seconds)
Evaluation Time: Xh Ym Zs (total seconds)
Total Pipeline:  Xh Ym Zs (total seconds)

Comparison with OpenCompass (4 days = 345600 seconds):
-------------------------------------------------------
OpenCompass: 345600 seconds (4 days)
This Pipeline: X seconds
Speedup: Yx faster
```

**Summary Files**:
- `.md`: Markdown summary with timing and results
- `.csv`: CSV with accuracy metrics and timing
- `.txt`: Plain text summary

---

## Manual Steps (If Not Using Full Pipeline)

### Option A: Run Inference Only

```bash
# Set subset (optional)
export HEALTHBENCH_SUBSET=dev_50

# Run inference
python run.py healthbench_infer_config.py --mode infer -w outputs/my_run
```

### Option B: Run Evaluation Only (on existing predictions)

```bash
# Base subset
python async_healthbench_eval.py \
    --predictions outputs/my_run/TIMESTAMP/predictions/MODEL_NAME \
    --dataset-subset base \
    --mode per_rubric \
    --concurrency 200 \
    --limit 50 \
    --subset dev_50 \
    --output results/healthbench.json \
    --judge-log judge_logs/healthbench.jsonl

# Hard subset
python async_healthbench_eval.py \
    --predictions outputs/my_run/TIMESTAMP/predictions/MODEL_NAME \
    --dataset-subset hard \
    --mode per_rubric \
    --concurrency 200 \
    --limit 25 \
    --subset dev_100 \
    --output results/healthbench_hard.json \
    --judge-log judge_logs/healthbench_hard.jsonl

# Consensus subset
python async_healthbench_eval.py \
    --predictions outputs/my_run/TIMESTAMP/predictions/MODEL_NAME \
    --dataset-subset consensus \
    --mode per_rubric \
    --concurrency 200 \
    --limit 25 \
    --subset dev_100 \
    --output results/healthbench_consensus.json \
    --judge-log judge_logs/healthbench_consensus.jsonl
```

---

## File Locations Reference

### Configuration Files
- **Inference Config**: `healthbench_infer_config.py`
- **Pipeline Script**: `run_healthbench_full.sh`
- **Async Evaluator**: `async_healthbench_eval.py`
- **Scoring Logic**: `healthbench_scoring.py`

### Dataset Files
- **Base**: `data/huihuixu/healthbench/2025-05-07-06-14-12_oss_eval.jsonl`
- **Hard**: `data/huihuixu/healthbench/hard_2025-05-08-21-00-10.jsonl`
- **Consensus**: `data/huihuixu/healthbench/consensus_2025-05-09-20-00-46.jsonl`

### Output Files (per run)
- **Predictions**: `outputs/RUN_NAME/TIMESTAMP/predictions/MODEL_NAME/`
- **Results**: `outputs/RUN_NAME/results/MODEL_NAME/*.json`
- **Logs**: `outputs/RUN_NAME/logs/eval/MODEL_NAME/*.out`
- **Summary**: `outputs/RUN_NAME/summary/summary_*.{md,csv,txt}`
- **Timing**: `outputs/RUN_NAME/timing.txt`
- **Judge Logs**: `outputs/RUN_NAME/judge_logs/*.jsonl`

---

## Key Features

### 1. Sharded Prediction Handling

Predictions may be split across multiple files:
- `healthbench_0.json`, `healthbench_1.json`, ..., `healthbench_99.json`
- The evaluator automatically:
  - Detects all shard files
  - Reindexes keys to avoid collisions
  - Warns if shards are missing

### 2. Resume Support

If evaluation is interrupted, resume from judge log:

```bash
python async_healthbench_eval.py \
    --predictions <pred_dir> \
    --dataset-subset base \
    --resume-from judge_logs/healthbench.jsonl \
    --judge-log judge_logs/healthbench.jsonl \
    --output results/healthbench.json
```

### 3. Timing Tracking

Pipeline automatically tracks:
- Inference time (OpenCompass)
- Evaluation time (async evaluator)
- Total pipeline time
- Comparison with OpenCompass baseline (4 days)

Saved to `timing.txt` in output directory.

---

## Troubleshooting

### Missing Prediction Files

**Symptom**: Warning about missing shards (e.g., shard 95)

**Cause**: OpenCompass inference failed for that shard (check `.out` log files)

**Solution**: Re-run inference for failed shards, or skip them (evaluation will continue with available predictions)

### API Errors

**Symptom**: Judge calls failing with 404/5xx errors

**Check**:
- Judge API env vars are set correctly
- API base URL is accessible
- Model name matches judge API's model list

**Solution**: The evaluator retries automatically, but persistent errors indicate configuration issues

### Prediction Count Mismatch

**Symptom**: Warning "Prediction count (X) != dataset count (Y)"

**Cause**: Inference didn't complete for all examples

**Solution**: Check inference logs, re-run inference if needed

### Missing Dataset Files

**Symptom**: FileNotFoundError when loading dataset

**Check**: Dataset files exist in `data/huihuixu/healthbench/`

**Solution**: Download HealthBench dataset files to correct location

---

## Performance Expectations

**OpenCompass Baseline**: ~4 days for full dataset

**Async Evaluator**:
- **Throughput**: 2+ rps (vs OC's ~0.21 rps)
- **Speedup**: ~10x faster evaluation
- **Total Pipeline**: Depends on inference time + evaluation time

**Example** (full dataset):
- Inference: ~X hours (depends on model/server)
- Evaluation: ~Y hours (depends on judge API capacity)
- **Total**: Much faster than 4 days

---

## Validation

Results are **identical** to OpenCompass because:
- ✅ Judge prompts are byte-for-byte identical (see `ASYNC_VS_OPENCOMPASS_EQUIVALENCE.md`)
- ✅ Scoring logic is functionally identical
- ✅ Data loading is identical

See `ASYNC_VS_OPENCOMPASS_EQUIVALENCE.md` for complete proof.

---

## Quick Reference

**One-command full pipeline**:
```bash
./run_healthbench_full.sh MODEL_NAME full 200
```

**What it does**:
1. Runs OpenCompass inference on full dataset
2. Runs async evaluation on all three subsets
3. Generates results, summaries, and timing

**Output location**: `outputs/healthbench_full_TIMESTAMP/`

**Key files to check**:
- `timing.txt` - Pipeline timing and speedup
- `summary/summary_*.md` - Human-readable summary
- `results/MODEL_NAME/*.json` - Detailed results per subset

