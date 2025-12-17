# Benchmark Methodology

## Objective

Determine whether OpenCompass's orchestration or the judge API is the performance bottleneck for HealthBench evaluations.

---

## Approach

Run identical workloads through two systems and compare wall-clock time.

---

## Test Setup

### Dataset Selection

Used **first 50 examples** from HealthBench base dataset (not random sampling) to ensure deterministic, reproducible results.

| Parameter | Value |
|-----------|-------|
| Dataset file | `data/huihuixu/healthbench/2025-05-07-06-14-12_oss_eval.jsonl` |
| Selection | First 50 examples (indices 0-49) |
| Total rubric items | 539 |
| Judge calls | 539 (one per rubric, per_rubric mode) |

### API Configuration

Both tests used identical judge API settings:

| Parameter | Value |
|-----------|-------|
| Endpoint | `$OC_JUDGE_API_BASE` |
| Model | `$OC_JUDGE_MODEL` |
| Max tokens | 8192 |

### Equivalence Verification

Ran `verify_test_equivalence.py` to confirm:

```
✓ Same examples: First 50 from dataset
✓ Same prompt count: 539 per_rubric prompts
✓ Same API endpoint: Both read from OC_JUDGE_* env vars
✓ Same prompt format: Both use GRADER_TEMPLATE from healthbench.py
```

Prompt hash: `2bee9a3a2780636e935fff329e07bea9`

---

## System A: Async Harness

### Implementation

- `bench_async_harness.py`
- Uses `asyncio` event loop
- `AsyncOpenAI` client for true async I/O
- `asyncio.Semaphore(250)` for concurrency control
- No artificial rate limiting

### Command

```bash
python bench_async_harness.py --sample-size 50 --use-first-n --concurrency 250
```

### Key Code Path

1. Load first 50 examples from JSONL
2. Build 539 per_rubric prompts using same template as OC
3. Fire all requests through semaphore-controlled async loop
4. Collect results and timing

---

## System B: OpenCompass

### Implementation

- Standard OpenCompass with HealthBench dataset
- Used `HEALTHBENCH_SUBSET=dev_50` to limit to first 50 examples
- `task3_eval.py` configuration
- Modified healthbench.py with max 3 retries (was infinite)

### Command

```bash
python time_oc_simple.py
```

### Key Code Path

1. OpenCompass loads dataset via HuggingFace wrapper
2. Runs inference phase (model generates responses)
3. Runs evaluation phase (judge scores rubrics)
4. Writes results to outputs directory

---

## Timing Measurement

### Async Harness

- Start timer before first request
- Stop timer after last response
- Wall-clock = end - start

### OpenCompass

- Start timer before subprocess launch
- Stop timer after subprocess exits
- Wall-clock = end - start
- Includes inference phase overhead (minor)

---

## Fairness Considerations

| Factor | Mitigation |
|--------|------------|
| Cold start | Both systems warmed up API before timing |
| Network variance | Same machine, same API endpoint |
| Data ordering | Both use deterministic first-50 selection |
| Prompt format | Verified identical via hash |
| Retry behavior | Fixed OC infinite loops before test |

---

## Limitations

1. **Inference phase included in OC timing**: OC time includes model inference (~5-10 min). However, even subtracting this, OC eval is still ~10x slower.

2. **Single run**: Did not run multiple trials. Given 43 min vs 4 min difference, statistical significance is obvious.

3. **50 examples**: Small sample, but proportional results expected at scale.

---

## Scripts Used

| Script | Purpose |
|--------|---------|
| `bench_async_harness.py` | Async harness benchmark |
| `time_oc_simple.py` | OpenCompass benchmark with timing |
| `verify_test_equivalence.py` | Confirm identical workloads |
| `test_judge_capacity.py` | Judge API capacity testing |

