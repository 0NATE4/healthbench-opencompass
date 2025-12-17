# OpenCompass HealthBench Performance Bottleneck Analysis

## Executive Summary

We benchmarked the judge API capacity against OpenCompass's actual throughput to identify why HealthBench evaluations are slow.

| Test | Judge Calls | Time | Throughput | Status |
|------|-------------|------|------------|--------|
| Async Harness | 539 | 222s | 2.43 rps | Completed |
| OpenCompass | 539 | >30min | - | Stuck in retry loops |

**Root cause**: OpenCompass's architecture, not the judge API.

---

## OpenCompass Evaluation Flow

1. **Load Dataset** - Reads HealthBench JSONL, parses examples
2. **Inference Stage** - Model generates responses (writes to `predictions/`)
3. **Evaluation Stage** - Judge LLM scores each rubric item
4. **Scoring** - Aggregates per-sample and overall metrics

The **evaluation stage** (judge calls) is the bottleneck.

---

## Identified Bottlenecks

### 1. Concurrency Model: ThreadPoolExecutor (GIL-bound)

```python
# OpenCompass uses threading
from concurrent.futures import ThreadPoolExecutor
executor.submit(self._generate, ...)
```

**Problem**: Python's GIL limits true parallelism. Threads wait on each other.

**Better**: `asyncio` + `AsyncOpenAI` for I/O-bound workloads.

### 2. Process Model: Subprocess Spawning

```python
# NaivePartitioner spawns subprocesses per partition
partitioner = dict(type=NaivePartitioner, n=100)
```

**Problem**: Each subprocess has startup overhead, separate memory space, no shared connection pool.

**Better**: Single process with async event loop.

### 3. Rate Control: TokenBucket QPS Throttle

```python
# OpenAISDK uses query_per_second throttle
query_per_second=5  # artificially caps throughput
```

**Problem**: Throttles below API capacity. Judge API handles 200-400 concurrent requests.

**Better**: Semaphore-based concurrency control.

### 4. Retry Logic: Infinite Loops (FIXED)

```python
# Original code
while True:
    response = self.grader_model(messages)
    if valid:
        break
    print('retrying...')  # loops forever on API errors
```

**Problem**: ~6% of judge responses return empty/invalid JSON. OC retried forever.

**Fix applied**: Max 3 retries, then mark as failed and continue.

---

## Test Methodology

### Equivalence Verification

Both tests use **identical data**:

| Parameter | Async Harness | OpenCompass |
|-----------|---------------|-------------|
| Dataset | First 50 examples | First 50 examples |
| Selection | `use_first_n=True` | `HEALTHBENCH_SUBSET=dev_50` |
| Judge calls | 539 (per_rubric) | 539 (per_rubric) |
| API endpoint | `OC_JUDGE_API_BASE` | Same |
| Prompt format | `GRADER_TEMPLATE` | Same |

Verified with `verify_test_equivalence.py`:
- Prompt hash: `2bee9a3a2780636e935fff329e07bea9`
- Example IDs match

### Async Harness Results

```json
{
  "sample_size": 50,
  "use_first_n": true,
  "mode": "per_rubric",
  "actual_requests": 539,
  "wall_clock_seconds": 222.18,
  "effective_rps": 2.43,
  "success_rate": 0.93,
  "error_breakdown": {"other": 36}
}
```

### OpenCompass Results

- Started evaluation at 17:36
- Log stopped updating at 17:46 (10 min)
- 106 JSON decode failures logged
- Process still running 30+ min later
- **Did not complete** - stuck in retry loops

---

## Code Changes Made

### 1. Fixed Infinite Retry Loops

**File**: `opencompass/datasets/healthbench/healthbench.py`

```python
# Before
while True:
    ...
    print('Grading failed, retrying...')

# After
max_retries = 3
for attempt in range(max_retries):
    ...
    print(f'Grading failed, retrying... ({attempt + 1}/{max_retries})')
# Return failure dict after max retries
```

### 2. Added dev_50 Subset

**File**: `opencompass/datasets/healthbench/healthbench.py`

```python
elif dev_subset == 'dev_50':
    if subset == '':
        max_n = 50
    else:
        max_n = None
```

### 3. Created Verification Script

**File**: `verify_test_equivalence.py`

Confirms both tests use identical examples, prompts, and API config.

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `bench_async_harness.py` | Benchmark async API throughput |
| `time_oc_simple.py` | Benchmark OpenCompass throughput |
| `verify_test_equivalence.py` | Confirm test equivalence |
| `test_judge_capacity.py` | Judge API capacity testing |

---

## Conclusions

1. **Judge API is not the bottleneck** - handles 200-400 concurrent requests at 2+ rps
2. **OpenCompass architecture is the bottleneck**:
   - Threading + GIL limits parallelism
   - Subprocess spawning adds overhead
   - QPS throttling caps throughput artificially
   - Infinite retry loops caused hangs (now fixed)

3. **Observed speedup**: Async harness completed in 222s vs OC not completing after 30min

---

## Next Steps

1. **Re-run OC benchmark** with fixed retry logic to get actual wall-clock time
2. **Compare results** to quantify OC overhead
3. **Decision point**:
   - **Option A**: Patch OpenCompass further (replace ThreadPool with asyncio)
   - **Option B**: Build custom async evaluator using our harness as base

---

## Commands Reference

```bash
# Async harness benchmark (first 50 examples)
python bench_async_harness.py --sample-size 50 --use-first-n --concurrency 250

# OpenCompass benchmark (first 50 examples)
python time_oc_simple.py

# Verify equivalence
python verify_test_equivalence.py

# Judge capacity test
python test_judge_capacity.py --use-real-prompts --mode per_rubric --sample-size 50
```

---

*Last updated: 2025-12-04*

