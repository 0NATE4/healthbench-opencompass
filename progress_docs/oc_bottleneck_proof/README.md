# OpenCompass Bottleneck Proof

This directory contains evidence proving that OpenCompass's orchestration overhead is the primary bottleneck for HealthBench LLM-judge evaluations.

## Key Finding

**OpenCompass is 12.3x slower than a simple async harness for identical workloads.**

## Contents

| File | Description |
|------|-------------|
| `proof_summary.md` | Executive summary with benchmark results |
| `methodology.md` | How the comparison was conducted |
| `prompt_comparison_appendix.md` | **Concrete prompt samples and equivalence proof** |
| `bottleneck_analysis.md` | Detailed breakdown of OC bottlenecks |
| `time_oc_results.json` | OpenCompass benchmark raw data |
| `bench_async_results.json` | Async harness benchmark raw data |
| `prompt_samples.json` | Full prompt samples (shortest & longest) |

## Quick Stats

| Metric | Async Harness | OpenCompass | Ratio |
|--------|---------------|-------------|-------|
| Time | 222s | 2591s | 12.3x |
| Throughput | 2.43 rps | 0.21 rps | 0.08x |
| Judge calls | 539 | 539 | 1:1 |

## Conclusion

The judge API is not the bottleneck. OpenCompass's architecture (threading, subprocesses, QPS throttling) wastes 92% of available capacity.

