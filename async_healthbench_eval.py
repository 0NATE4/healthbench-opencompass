#!/usr/bin/env python3
"""Async HealthBench evaluator - runs judge calls at high throughput.

Replaces OpenCompass's slow evaluation stage with async API calls.
Achieves 2+ rps vs OC's ~0.21 rps.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, APIStatusError

from opencompass.utils.text_postprocessors import extract_non_reasoning_content

from healthbench_scoring import (
    RubricItem,
    ScoredExample,
    aggregate_scores,
    build_grader_prompt_per_example,
    build_grader_prompt_per_rubric,
    calculate_score,
    parse_json_response,
    parse_per_example_response,
)

# Timeout exceptions
try:
    from httpx import ReadTimeout, ConnectTimeout
    HTTPX_TIMEOUTS = (ReadTimeout, ConnectTimeout)
except ImportError:
    HTTPX_TIMEOUTS = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async HealthBench evaluator - fast judge calls"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to OC predictions JSON file or directory of sharded files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HealthBench JSONL dataset file (auto-set if --dataset-subset used)",
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        choices=["base", "hard", "consensus"],
        default=None,
        help="HealthBench subset: base, hard, or consensus. Auto-selects dataset file and prediction pattern.",
    )
    parser.add_argument(
        "--mode",
        choices=["per_rubric", "per_example"],
        default="per_rubric",
        help="Judge mode: per_rubric (one call per rubric) or per_example (one call per example)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=200,
        help="Max concurrent judge API calls (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="async_eval_results.json",
        help="Path to write final results JSON",
    )
    parser.add_argument(
        "--judge-log",
        type=str,
        default=None,
        help="Path to streaming JSONL of judge outputs (enables resume)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing JSONL to resume from (skip already-judged items)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Judge API base URL (default: $OC_JUDGE_API_BASE)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Judge API key (default: $OC_JUDGE_API_KEY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Judge model name (default: $OC_JUDGE_MODEL)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens for judge response",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per judge call",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit to first N examples (0 = no limit)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Subset name for metadata (e.g., dev_50, dev_tiny, full)",
    )
    return parser.parse_args()


import re

# Dataset file mapping
DATASET_FILES = {
    "base": "2025-05-07-06-14-12_oss_eval.jsonl",
    "hard": "hard_2025-05-08-21-00-10.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
}

# Prediction file patterns per subset
# All subsets can be sharded (healthbench_0.json, healthbench_hard_0.json, etc.)
PREDICTION_PATTERNS = {
    "base": r"^healthbench(_\d+)?\.json$",
    "hard": r"^healthbench_hard(_\d+)?\.json$",
    "consensus": r"^healthbench_consensus(_\d+)?\.json$",
}


def _extract_shard_number(filename: str) -> int:
    """Extract shard number from filename like healthbench_0.json, healthbench_hard_1.json, etc.
    
    Handles patterns:
    - healthbench.json -> 0
    - healthbench_0.json -> 0
    - healthbench_hard.json -> 0
    - healthbench_hard_0.json -> 0
    - healthbench_consensus_1.json -> 1
    """
    # Match base, hard, or consensus with optional shard number
    match = re.match(r'healthbench(?:_(?:hard|consensus))?(?:_(\d+))?\.json$', filename)
    if match and match.group(1):
        return int(match.group(1))
    return 0


def load_predictions(path: str, dataset_subset: str | None = None) -> dict[str, dict[str, Any]]:
    """Load predictions from JSON file or directory of sharded files.
    
    Args:
        path: Path to predictions file or directory
        dataset_subset: If set, filter files to only those matching the subset pattern
        
    Returns dict mapping string index ("0", "1", ...) to prediction record.
    
    Note: When merging sharded files (healthbench_0.json, healthbench_1.json, etc.),
    keys are reindexed based on cumulative offset to avoid overwrites.
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        with open(path_obj, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    if path_obj.is_dir():
        # Get pattern for filtering
        pattern = None
        if dataset_subset and dataset_subset in PREDICTION_PATTERNS:
            pattern = re.compile(PREDICTION_PATTERNS[dataset_subset])
        
        # Collect matching JSON files
        matched_files = []
        for json_file in path_obj.glob("*.json"):
            # Filter by pattern if specified
            if pattern and not pattern.match(json_file.name):
                continue
            matched_files.append(json_file)
        
        # Fail loudly if no files matched
        if dataset_subset:
            if not matched_files:
                print(f"\n❌ ERROR: No prediction files matched pattern for '{dataset_subset}'", file=sys.stderr)
                print(f"   Pattern: {PREDICTION_PATTERNS[dataset_subset]}", file=sys.stderr)
                print(f"   Directory: {path}", file=sys.stderr)
                print(f"   Available files: {list(path_obj.glob('*.json'))}", file=sys.stderr)
                sys.exit(1)
        elif not matched_files:
            print(f"\n⚠️  WARNING: No JSON files found in {path}", file=sys.stderr)
            return {}
        
        # Sort files by shard number to ensure correct ordering
        matched_files.sort(key=lambda f: _extract_shard_number(f.name))
        
        # Merge with reindexing to avoid key collisions
        merged: dict[str, dict[str, Any]] = {}
        offset = 0
        shard_numbers = []
        
        for json_file in matched_files:
            shard_num = _extract_shard_number(json_file.name)
            shard_numbers.append(shard_num)
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reindex keys based on cumulative offset
            reindexed = {str(int(k) + offset): v for k, v in data.items()}
            merged.update(reindexed)
            
            print(f"  Shard {shard_num} ({json_file.name}): {len(data)} predictions, offset={offset}")
            offset += len(data)
        
        # Check for missing shards (gaps in sequence)
        if shard_numbers:
            min_shard = min(shard_numbers)
            max_shard = max(shard_numbers)
            expected_count = max_shard - min_shard + 1
            actual_count = len(matched_files)
            missing_shards = []
            for i in range(min_shard, max_shard + 1):
                if i not in shard_numbers:
                    missing_shards.append(i)
            
            if missing_shards:
                print(f"\n⚠️  WARNING: Missing shard(s): {missing_shards}", file=sys.stderr)
                print(f"   Expected {expected_count} shards (range {min_shard}-{max_shard}), found {actual_count}", file=sys.stderr)
        
        # Log summary
        file_names = [f.name for f in matched_files]
        if shard_numbers:
            shard_range = f" (shards {min(shard_numbers)}-{max(shard_numbers)})"
        else:
            shard_range = ""
        print(f"Loaded {len(merged)} predictions from {len(matched_files)} shard(s){shard_range}")
        
        return merged
    
    raise FileNotFoundError(f"Predictions path not found: {path}")


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load HealthBench dataset from JSONL."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def load_resume_state(path: str) -> dict[str, set[int]]:
    """Load already-judged items from JSONL.
    
    Returns dict mapping prompt_id to set of judged rubric indices.
    """
    if not path or not os.path.exists(path):
        return {}
    
    judged: dict[str, set[int]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                prompt_id = record.get('prompt_id', '')
                rubric_idx = record.get('rubric_index', -1)
                if prompt_id not in judged:
                    judged[prompt_id] = set()
                judged[prompt_id].add(rubric_idx)
    return judged


class JudgeLogWriter:
    """Streaming JSONL writer for judge results."""
    
    def __init__(self, path: str | None):
        self.path = path
        self.file = None
        if path:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            self.file = open(path, 'a', encoding='utf-8')
    
    def write(self, record: dict[str, Any]) -> None:
        if self.file:
            self.file.write(json.dumps(record) + '\n')
            self.file.flush()
    
    def close(self) -> None:
        if self.file:
            self.file.close()


async def judge_call(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    max_retries: int,
) -> tuple[str | None, str | None]:
    """Make a single judge API call with retries.
    
    Returns (response_text, error_message).
    """
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = resp.choices[0].message.content or ""
            if text:
                return text, None
            return None, "Empty response"
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            return None, "Timeout after retries"
        except APIStatusError as e:
            status = e.response.status_code
            # Retry on rate limits (429) and transient errors (404, 5xx)
            if status in (429, 404, 500, 502, 503, 504) and attempt < max_retries - 1:
                backoff = 2 ** attempt
                if status == 429:
                    backoff = min(backoff, 10)  # Cap rate limit backoff
                # Log retries for debugging (but not too verbose)
                if attempt == 0:  # Only log first retry attempt
                    print(f"  Retrying after {status} error (attempt {attempt + 1}/{max_retries})", flush=True)
                await asyncio.sleep(backoff)
                continue
            return None, f"API error: {status}"
        except Exception as e:
            if HTTPX_TIMEOUTS and isinstance(e, HTTPX_TIMEOUTS):
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
            return None, f"Error: {str(e)}"
    return None, "Max retries exceeded"


async def run_per_rubric_evaluation(
    client: AsyncOpenAI,
    model: str,
    predictions: dict[str, dict[str, Any]],
    dataset: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    resume_state: dict[str, set[int]],
    log_writer: JudgeLogWriter,
    verbose: bool,
) -> list[ScoredExample]:
    """Run evaluation in per_rubric mode (one judge call per rubric item)."""
    
    sem = asyncio.Semaphore(concurrency)
    scored_examples: list[ScoredExample] = []
    
    total_calls = 0
    completed_calls = 0
    start_time = time.time()
    
    # Count total calls needed
    for idx, example in enumerate(dataset):
        prompt_id = example.get('prompt_id', f'example_{idx}')
        rubrics = example.get('rubrics', [])
        already_judged = resume_state.get(prompt_id, set())
        for ri in range(len(rubrics)):
            if ri not in already_judged:
                total_calls += 1
    
    print(f"Total judge calls needed: {total_calls}")
    
    async def process_rubric(
        example_idx: int,
        example: dict[str, Any],
        rubric_idx: int,
        rubric_data: dict[str, Any],
        prediction: str,
        conversation: list[dict[str, str]],
    ) -> dict[str, Any]:
        nonlocal completed_calls
        
        rubric_item = RubricItem.from_dict(rubric_data)
        prompt = build_grader_prompt_per_rubric(conversation, prediction, rubric_item)
        
        async with sem:
            response_text, error = await judge_call(
                client, model, prompt, max_tokens, timeout, max_retries
            )
        
        completed_calls += 1
        if completed_calls % 100 == 0 or completed_calls == total_calls:
            elapsed = time.time() - start_time
            rps = completed_calls / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed_calls}/{total_calls} ({rps:.2f} rps)")
        
        prompt_id = example.get('prompt_id', f'example_{example_idx}')
        
        if response_text:
            parsed = parse_json_response(response_text)
            criteria_met = parsed.get('criteria_met')
            explanation = parsed.get('explanation', 'No explanation')
        else:
            criteria_met = None
            explanation = error or 'Unknown error'
        
        result = {
            'prompt_id': prompt_id,
            'rubric_index': rubric_idx,
            'rubric_points': rubric_item.points,
            'criteria_met': criteria_met,
            'explanation': explanation,
            'raw_response': response_text,
        }
        
        log_writer.write(result)
        
        return {
            'example_idx': example_idx,
            'rubric_idx': rubric_idx,
            **rubric_item.to_dict(),
            'criteria_met': criteria_met,
            'explanation': explanation,
        }
    
    # Build list of tasks
    tasks = []
    for idx, example in enumerate(dataset):
        str_idx = str(idx)
        if str_idx not in predictions:
            if verbose:
                print(f"Warning: No prediction for index {idx}")
            continue
        
        pred_record = predictions[str_idx]
        prediction_raw = pred_record.get('prediction', '')
        # Apply postprocessor to extract actual response (removes reasoning tags)
        prediction = extract_non_reasoning_content(prediction_raw)
        conversation = pred_record.get('origin_prompt', example.get('prompt', []))
        
        prompt_id = example.get('prompt_id', f'example_{idx}')
        already_judged = resume_state.get(prompt_id, set())
        
        rubrics = example.get('rubrics', [])
        for ri, rubric_data in enumerate(rubrics):
            if ri in already_judged:
                continue
            tasks.append(process_rubric(idx, example, ri, rubric_data, prediction, conversation))
    
    # Run all tasks
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Group results by example
    example_results: dict[int, list[dict[str, Any]]] = {}
    for r in all_results:
        if isinstance(r, Exception):
            print(f"Task exception: {r}")
            continue
        ex_idx = r['example_idx']
        if ex_idx not in example_results:
            example_results[ex_idx] = []
        example_results[ex_idx].append(r)
    
    # Also load any existing results from resume state
    for prompt_id, judged_indices in resume_state.items():
        # Find example index for this prompt_id
        for idx, ex in enumerate(dataset):
            if ex.get('prompt_id', f'example_{idx}') == prompt_id:
                # We need to reconstruct results from the resume file
                # For now, we skip this - the resume file should be reloaded
                break
    
    # Build ScoredExample objects
    for idx, example in enumerate(dataset):
        prompt_id = example.get('prompt_id', f'example_{idx}')
        rubrics = example.get('rubrics', [])
        example_tags = example.get('example_tags', [])
        
        # Get results for this example
        results = example_results.get(idx, [])
        
        # Sort by rubric index
        results.sort(key=lambda x: x['rubric_idx'])
        
        # If we don't have all rubrics, skip scoring
        if len(results) != len(rubrics):
            if verbose:
                print(f"Incomplete results for {prompt_id}: {len(results)}/{len(rubrics)}")
            continue
        
        # Calculate score
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        score = calculate_score(rubric_items, results)
        
        if score is not None:
            scored_examples.append(ScoredExample(
                prompt_id=prompt_id,
                example_index=idx,
                overall_score=score,
                example_tags=example_tags,
                rubric_results=results,
            ))
    
    return scored_examples


async def run_per_example_evaluation(
    client: AsyncOpenAI,
    model: str,
    predictions: dict[str, dict[str, Any]],
    dataset: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
    timeout: float,
    max_retries: int,
    resume_state: dict[str, set[int]],
    log_writer: JudgeLogWriter,
    verbose: bool,
) -> list[ScoredExample]:
    """Run evaluation in per_example mode (one judge call per example)."""
    
    sem = asyncio.Semaphore(concurrency)
    scored_examples: list[ScoredExample] = []
    
    total_calls = 0
    completed_calls = 0
    start_time = time.time()
    
    # Count total calls (skip fully-judged examples)
    for idx, example in enumerate(dataset):
        prompt_id = example.get('prompt_id', f'example_{idx}')
        rubrics = example.get('rubrics', [])
        already_judged = resume_state.get(prompt_id, set())
        # In per_example mode, if we have all rubrics judged, skip
        if len(already_judged) < len(rubrics):
            total_calls += 1
    
    print(f"Total judge calls needed: {total_calls}")
    
    async def process_example(
        example_idx: int,
        example: dict[str, Any],
        prediction: str,
        conversation: list[dict[str, str]],
    ) -> tuple[int, list[dict[str, Any]]]:
        nonlocal completed_calls
        
        rubrics = example.get('rubrics', [])
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        
        prompt = build_grader_prompt_per_example(conversation, prediction, rubric_items)
        
        async with sem:
            response_text, error = await judge_call(
                client, model, prompt, max_tokens, timeout, max_retries
            )
        
        completed_calls += 1
        if completed_calls % 50 == 0 or completed_calls == total_calls:
            elapsed = time.time() - start_time
            rps = completed_calls / elapsed if elapsed > 0 else 0
            print(f"Progress: {completed_calls}/{total_calls} ({rps:.2f} rps)")
        
        prompt_id = example.get('prompt_id', f'example_{example_idx}')
        
        if response_text:
            parsed_results = parse_per_example_response(response_text, len(rubric_items))
        else:
            parsed_results = [
                {'criteria_met': None, 'explanation': error or 'Unknown error'}
                for _ in rubric_items
            ]
        
        # Log each rubric result
        results = []
        for ri, (rubric_item, parsed) in enumerate(zip(rubric_items, parsed_results)):
            result = {
                'prompt_id': prompt_id,
                'rubric_index': ri,
                'rubric_points': rubric_item.points,
                'criteria_met': parsed['criteria_met'],
                'explanation': parsed['explanation'],
                'raw_response': response_text if ri == 0 else None,  # Only store once
            }
            log_writer.write(result)
            results.append({
                **rubric_item.to_dict(),
                'criteria_met': parsed['criteria_met'],
                'explanation': parsed['explanation'],
            })
        
        return example_idx, results
    
    # Build list of tasks
    tasks = []
    for idx, example in enumerate(dataset):
        str_idx = str(idx)
        if str_idx not in predictions:
            if verbose:
                print(f"Warning: No prediction for index {idx}")
            continue
        
        pred_record = predictions[str_idx]
        prediction_raw = pred_record.get('prediction', '')
        # Apply postprocessor to extract actual response (removes reasoning tags)
        prediction = extract_non_reasoning_content(prediction_raw)
        conversation = pred_record.get('origin_prompt', example.get('prompt', []))
        
        prompt_id = example.get('prompt_id', f'example_{idx}')
        already_judged = resume_state.get(prompt_id, set())
        rubrics = example.get('rubrics', [])
        
        # Skip if fully judged
        if len(already_judged) >= len(rubrics):
            continue
        
        tasks.append(process_example(idx, example, prediction, conversation))
    
    # Run all tasks
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Group results by example
    example_results: dict[int, list[dict[str, Any]]] = {}
    for r in all_results:
        if isinstance(r, Exception):
            print(f"Task exception: {r}")
            continue
        ex_idx, results = r
        example_results[ex_idx] = results
    
    # Build ScoredExample objects
    for idx, example in enumerate(dataset):
        prompt_id = example.get('prompt_id', f'example_{idx}')
        rubrics = example.get('rubrics', [])
        example_tags = example.get('example_tags', [])
        
        results = example_results.get(idx, [])
        
        if len(results) != len(rubrics):
            if verbose:
                print(f"Incomplete results for {prompt_id}: {len(results)}/{len(rubrics)}")
            continue
        
        rubric_items = [RubricItem.from_dict(r) for r in rubrics]
        score = calculate_score(rubric_items, results)
        
        if score is not None:
            scored_examples.append(ScoredExample(
                prompt_id=prompt_id,
                example_index=idx,
                overall_score=score,
                example_tags=example_tags,
                rubric_results=results,
            ))
    
    return scored_examples


async def main_async() -> None:
    args = parse_args()
    
    # Resolve API config
    api_base = args.api_base or os.environ.get("OC_JUDGE_API_BASE")
    api_key = args.api_key or os.environ.get("OC_JUDGE_API_KEY")
    model = args.model or os.environ.get("OC_JUDGE_MODEL")
    
    if not api_base:
        print("Error: Missing API base (set OC_JUDGE_API_BASE or --api-base)", file=sys.stderr)
        sys.exit(1)
    if not api_key:
        print("Error: Missing API key (set OC_JUDGE_API_KEY or --api-key)", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: Missing model (set OC_JUDGE_MODEL or --model)", file=sys.stderr)
        sys.exit(1)
    
    # Resolve dataset path from dataset_subset if not provided
    dataset_path = args.dataset
    dataset_subset = args.dataset_subset
    
    if dataset_subset and not dataset_path:
        # Auto-resolve dataset path based on subset
        if dataset_subset not in DATASET_FILES:
            print(f"Error: Unknown dataset subset '{dataset_subset}'", file=sys.stderr)
            sys.exit(1)
        # Look for dataset in standard location
        base_dir = Path("data/huihuixu/healthbench")
        dataset_path = str(base_dir / DATASET_FILES[dataset_subset])
        print(f"Auto-resolved dataset path: {dataset_path}")
    
    if not dataset_path:
        print("Error: Must provide --dataset or --dataset-subset", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("Async HealthBench Evaluator")
    print("=" * 60)
    print(f"Predictions: {args.predictions}")
    print(f"Dataset: {dataset_path}")
    print(f"Dataset Subset: {dataset_subset or 'Not specified'}")
    print(f"Mode: {args.mode}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Limit: {args.limit if args.limit > 0 else 'None'}")
    print(f"Dev Subset: {args.subset or 'Not specified'}")
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Judge Log: {args.judge_log or 'None'}")
    print(f"Resume From: {args.resume_from or 'None'}")
    print("=" * 60)
    
    # Load data
    print("\nLoading predictions...")
    predictions = load_predictions(args.predictions, dataset_subset)
    print(f"Loaded {len(predictions)} predictions")
    
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} examples from dataset")
    
    # Apply limit if specified
    if args.limit > 0:
        dataset = dataset[:args.limit]
        print(f"Applied --limit: using first {len(dataset)} examples")
    
    # Safety checks for alignment
    num_preds = len(predictions)
    num_examples = len(dataset)
    
    if num_preds != num_examples:
        print(f"\n⚠️  WARNING: Prediction count ({num_preds}) != dataset count ({num_examples})")
        if num_preds < num_examples:
            print(f"   Only {num_preds} examples will be evaluated (missing predictions for rest)")
        else:
            print(f"   Extra predictions will be ignored")
    
    # Verify we have predictions for at least some examples
    matched = sum(1 for i in range(num_examples) if str(i) in predictions)
    if matched == 0:
        print("\n❌ ERROR: No predictions match dataset indices!", file=sys.stderr)
        print("   Check that predictions use '0', '1', '2'... as keys", file=sys.stderr)
        sys.exit(1)
    
    # Log alignment verification
    print(f"\n📋 Alignment check ({matched}/{num_examples} examples have predictions):")
    for i in range(min(5, num_examples)):
        str_idx = str(i)
        prompt_id = dataset[i].get('prompt_id', 'N/A')
        has_pred = str_idx in predictions
        status = "✓" if has_pred else "✗"
        print(f"  [{i}] {status} prompt_id={prompt_id}")
    if num_examples > 5:
        print(f"  ... ({num_examples - 5} more)")
    print()
    
    # Load resume state
    resume_state = load_resume_state(args.resume_from)
    if resume_state:
        total_resumed = sum(len(v) for v in resume_state.values())
        print(f"Resuming: {total_resumed} rubrics already judged across {len(resume_state)} examples")
    
    # Setup
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    log_writer = JudgeLogWriter(args.judge_log)
    
    start_time = time.time()
    
    try:
        if args.mode == "per_rubric":
            scored_examples = await run_per_rubric_evaluation(
                client=client,
                model=model,
                predictions=predictions,
                dataset=dataset,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                resume_state=resume_state,
                log_writer=log_writer,
                verbose=args.verbose,
            )
        else:
            scored_examples = await run_per_example_evaluation(
                client=client,
                model=model,
                predictions=predictions,
                dataset=dataset,
                concurrency=args.concurrency,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                max_retries=args.max_retries,
                resume_state=resume_state,
                log_writer=log_writer,
                verbose=args.verbose,
            )
    finally:
        log_writer.close()
    
    elapsed = time.time() - start_time
    
    # Aggregate scores
    print("\n" + "=" * 60)
    print("Aggregating scores...")
    metrics = aggregate_scores(scored_examples)
    
    # Count judge calls attempted vs succeeded
    total_judge_calls = 0
    successful_judge_calls = 0
    failed_judge_calls = 0
    
    # Count from scored examples (each example has multiple rubrics)
    for ex in scored_examples:
        for result in ex.rubric_results:
            total_judge_calls += 1
            if result.get('criteria_met') is not None:
                successful_judge_calls += 1
            else:
                failed_judge_calls += 1
    
    # Add config to output (clearly show subset and limits)
    output = {
        **metrics,
        'config': {
            'mode': args.mode,
            'concurrency': args.concurrency,
            'judge_model': model,
            'max_tokens': args.max_tokens,
            'timeout': args.timeout,
            'limit': args.limit if args.limit > 0 else None,
            'dev_subset': args.subset,
            'dataset_subset': dataset_subset,  # base, hard, or consensus
            'predictions_path': args.predictions,
            'dataset_path': dataset_path,
        },
        'timing': {
            'total_seconds': elapsed,
            'examples_scored': len(scored_examples),
            'rps': len(scored_examples) / elapsed if elapsed > 0 else 0,
        },
        'judge_stats': {
            'total_calls': total_judge_calls,
            'successful': successful_judge_calls,
            'failed': failed_judge_calls,
            'success_rate': successful_judge_calls / total_judge_calls if total_judge_calls > 0 else 0,
        },
    }
    
    # Write results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if dataset_subset:
        print(f"Dataset Subset: {dataset_subset}")
    if args.subset:
        print(f"Dev Subset: {args.subset}")
    if args.limit > 0:
        print(f"Limit Applied: {args.limit} examples")
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Accuracy Std: {metrics['accuracy_std']:.4f}")
    print(f"N Samples: {metrics['n_samples']} (for {dataset_subset or 'dataset'})")
    print(f"\nTiming:")
    print(f"  Total Time: {elapsed:.1f}s")
    print(f"  Throughput: {output['timing']['rps']:.2f} rps")
    print(f"\nJudge Call Stats:")
    print(f"  Total Calls: {output['judge_stats']['total_calls']}")
    print(f"  Successful: {output['judge_stats']['successful']}")
    print(f"  Failed: {output['judge_stats']['failed']}")
    print(f"  Success Rate: {output['judge_stats']['success_rate']*100:.1f}%")
    print(f"\n📄 Results written to: {args.output}")
    if args.judge_log:
        print(f"📝 Judge log written to: {args.judge_log}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

