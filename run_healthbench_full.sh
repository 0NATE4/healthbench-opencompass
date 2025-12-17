#!/usr/bin/env bash
# run_healthbench_full.sh
# Full HealthBench pipeline: OC inference + async evaluation for all subsets
#
# Usage:
#   ./run_healthbench_full.sh MODEL_NAME [DEV_SUBSET] [CONCURRENCY]
#
# Examples:
#   ./run_healthbench_full.sh qwen3-4b-thinking-2507 dev_50 200
#   ./run_healthbench_full.sh my_model full 300
#   ./run_healthbench_full.sh my_model dev_tiny

set -euo pipefail

# Arguments
MODEL_NAME="${1:-}"
DEV_SUBSET="${2:-dev_50}"
CONCURRENCY="${3:-200}"

# Validate required args
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: $0 MODEL_NAME [DEV_SUBSET] [CONCURRENCY]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_NAME    Required. Model name/abbr as it appears in OC predictions dir"
    echo "  DEV_SUBSET    Optional. One of: dev_tiny, dev_50, dev_100, dev_medium, full (default: dev_50)"
    echo "  CONCURRENCY   Optional. Max concurrent judge calls (default: 200)"
    echo ""
    echo "Examples:"
    echo "  $0 qwen3-4b-thinking-2507 dev_50 200"
    echo "  $0 my_model full 300"
    exit 1
fi

# Determine limit based on dev subset
case "$DEV_SUBSET" in
    dev_tiny)
        LIMIT_BASE=2
        LIMIT_HARD=1
        LIMIT_CONSENSUS=2
        ;;
    dev_50)
        LIMIT_BASE=50
        LIMIT_HARD=0  # Skip hard/consensus for dev_50
        LIMIT_CONSENSUS=0
        ;;
    dev_100)
        LIMIT_BASE=50
        LIMIT_HARD=25
        LIMIT_CONSENSUS=25
        ;;
    dev_medium)
        LIMIT_BASE=250
        LIMIT_HARD=100
        LIMIT_CONSENSUS=300
        ;;
    full|"")
        LIMIT_BASE=0
        LIMIT_HARD=0
        LIMIT_CONSENSUS=0
        ;;
    *)
        echo "Warning: Unknown dev subset '$DEV_SUBSET', using no limit"
        LIMIT_BASE=0
        LIMIT_HARD=0
        LIMIT_CONSENSUS=0
        ;;
esac

# Paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/healthbench_${DEV_SUBSET}_${TIMESTAMP}"
DATA_DIR="data/huihuixu/healthbench"

# Track overall pipeline timing
PIPELINE_START_TIME=$(date +%s)

echo "============================================================"
echo "HealthBench Full Pipeline"
echo "============================================================"
echo "Model:       $MODEL_NAME"
echo "Dev Subset:  $DEV_SUBSET"
echo "Concurrency: $CONCURRENCY"
echo "Output Dir:  $OUTPUT_DIR"
echo "============================================================"
echo ""

# Step 1: Run OC inference
echo "[Step 1/2] Running OpenCompass inference..."
echo ""

INFERENCE_START_TIME=$(date +%s)

if [ "$DEV_SUBSET" != "full" ]; then
    export HEALTHBENCH_SUBSET="$DEV_SUBSET"
fi

python run.py \
    healthbench_infer_config.py \
    --mode infer \
    -w "$OUTPUT_DIR"

INFERENCE_END_TIME=$(date +%s)
INFERENCE_DURATION=$((INFERENCE_END_TIME - INFERENCE_START_TIME))
INFERENCE_HOURS=$((INFERENCE_DURATION / 3600))
INFERENCE_MINS=$(((INFERENCE_DURATION % 3600) / 60))
INFERENCE_SECS=$((INFERENCE_DURATION % 60))

echo ""
echo "[Step 1/2] Inference complete."
echo ""

# Step 2: Locate predictions directory
echo "Locating predictions directory..."

# OpenCompass may create a timestamped subdirectory inside work_dir when running inference-only
# Structure: work_dir/TIMESTAMP/predictions/MODEL/
# But for full runs, predictions are directly in work_dir/predictions/MODEL/
# Check both locations
TIMESTAMP_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)

if [ -n "$TIMESTAMP_DIR" ] && [ -d "$TIMESTAMP_DIR/predictions" ]; then
    # Inference-only run: predictions are in timestamp subdirectory
    echo "  Found timestamp directory: $(basename "$TIMESTAMP_DIR")"
    PRED_DIR="$TIMESTAMP_DIR/predictions/$MODEL_NAME"
else
    # Full run or direct structure: predictions are directly in work_dir
    echo "  Using direct structure (no timestamp subdirectory)"
    PRED_DIR="$OUTPUT_DIR/predictions/$MODEL_NAME"
fi

if [ ! -d "$PRED_DIR" ]; then
    # Try to find any model directory under predictions
    echo "  Model-specific directory not found, searching..."
    
    # Determine search location
    if [ -n "$TIMESTAMP_DIR" ] && [ -d "$TIMESTAMP_DIR/predictions" ]; then
        SEARCH_DIR="$TIMESTAMP_DIR/predictions"
    else
        SEARCH_DIR="$OUTPUT_DIR/predictions"
    fi
    
    echo "  Looking in: $SEARCH_DIR"
    
    # List what's actually there
    echo "  Contents:"
    ls -la "$SEARCH_DIR" 2>/dev/null || echo "   (empty or inaccessible)"
    
    FOUND_DIR=$(find "$SEARCH_DIR" -type d -mindepth 1 -maxdepth 1 2>/dev/null | head -1)
    if [ -n "$FOUND_DIR" ] && [ -d "$FOUND_DIR" ]; then
        PRED_DIR="$FOUND_DIR"
        echo "  ✓ Found model directory: $PRED_DIR"
    else
        echo "❌ ERROR: Could not find any model directory under predictions" >&2
        echo "   Expected: $PRED_DIR" >&2
        echo "   Searched in: $SEARCH_DIR" >&2
        echo "   Available directories:" >&2
        ls -la "$SEARCH_DIR" 2>/dev/null || echo "   (none)" >&2
        exit 1
    fi
else
    echo "  ✓ Found: $PRED_DIR"
fi

# Verify it has JSON files
JSON_COUNT=$(find "$PRED_DIR" -name "*.json" 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "⚠️  WARNING: No JSON files found in $PRED_DIR" >&2
    echo "   This may indicate inference didn't complete successfully." >&2
fi

echo "  JSON files found: $JSON_COUNT"
echo ""

# Step 3: Run async evaluator for each subset
echo "[Step 2/2] Running async evaluation..."
echo ""

EVALUATION_START_TIME=$(date +%s)

# Create directories matching OC structure
RESULTS_DIR="$OUTPUT_DIR/results/$MODEL_NAME"
mkdir -p "$RESULTS_DIR"
LOGS_EVAL_DIR="$OUTPUT_DIR/logs/eval/$MODEL_NAME"
mkdir -p "$LOGS_EVAL_DIR"
SUMMARY_DIR="$OUTPUT_DIR/summary"
mkdir -p "$SUMMARY_DIR"
JUDGE_LOG_DIR="$OUTPUT_DIR/judge_logs"  # Extra: OC doesn't have this, but useful for resume
mkdir -p "$JUDGE_LOG_DIR"

# Function to run evaluation for a subset
run_eval() {
    local SUBSET_NAME="$1"
    local LIMIT="$2"
    
    # Skip if limit is 0 and not full run
    if [ "$LIMIT" = "0" ] && [ "$DEV_SUBSET" != "full" ]; then
        echo "Skipping $SUBSET_NAME (limit=0 for $DEV_SUBSET)"
        return 0
    fi
    
    # Map subset name to OC's file naming
    case "$SUBSET_NAME" in
        base)
            OC_FILENAME="healthbench.json"
            ;;
        hard)
            OC_FILENAME="healthbench_hard.json"
            ;;
        consensus)
            OC_FILENAME="healthbench_consensus.json"
            ;;
        *)
            OC_FILENAME="healthbench_${SUBSET_NAME}.json"
            ;;
    esac
    
    local JUDGE_LOG="$JUDGE_LOG_DIR/${OC_FILENAME%.json}.jsonl"
    local RESULTS_FILE="$RESULTS_DIR/$OC_FILENAME"
    local EVAL_LOG="$LOGS_EVAL_DIR/${OC_FILENAME%.json}.out"
    
    echo "--- Evaluating: $SUBSET_NAME (limit=$LIMIT) ---"
    
    # Run evaluator, redirecting stdout/stderr to eval log (matching OC)
    python async_healthbench_eval.py \
        --predictions "$PRED_DIR" \
        --dataset-subset "$SUBSET_NAME" \
        --mode per_rubric \
        --concurrency "$CONCURRENCY" \
        --limit "$LIMIT" \
        --subset "$DEV_SUBSET" \
        --output "$RESULTS_FILE" \
        --judge-log "$JUDGE_LOG" \
        2>&1 | tee "$EVAL_LOG"
    
    echo "Results: $RESULTS_FILE"
    echo "Eval Log: $EVAL_LOG"
    echo ""
}

# Run for each subset
run_eval "base" "$LIMIT_BASE"
run_eval "hard" "$LIMIT_HARD"
run_eval "consensus" "$LIMIT_CONSENSUS"

EVALUATION_END_TIME=$(date +%s)
EVALUATION_DURATION=$((EVALUATION_END_TIME - EVALUATION_START_TIME))
EVALUATION_HOURS=$((EVALUATION_DURATION / 3600))
EVALUATION_MINS=$(((EVALUATION_DURATION % 3600) / 60))
EVALUATION_SECS=$((EVALUATION_DURATION % 60))

PIPELINE_END_TIME=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END_TIME - PIPELINE_START_TIME))
PIPELINE_HOURS=$((PIPELINE_DURATION / 3600))
PIPELINE_MINS=$(((PIPELINE_DURATION % 3600) / 60))
PIPELINE_SECS=$((PIPELINE_DURATION % 60))

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Output Dir: $OUTPUT_DIR"
echo "Results: $RESULTS_DIR"
echo "Eval Logs: $LOGS_EVAL_DIR"
echo "Summary: $SUMMARY_DIR"
echo "Judge Logs: $JUDGE_LOG_DIR (extra, for resume)"
echo ""
# Write timing to file
TIMING_FILE="$OUTPUT_DIR/timing.txt"
cat > "$TIMING_FILE" <<EOF
HealthBench Pipeline Timing
============================
Model: $MODEL_NAME
Dev Subset: $DEV_SUBSET
Concurrency: $CONCURRENCY
Timestamp: $(basename "$OUTPUT_DIR")

Inference Time:  ${INFERENCE_HOURS}h ${INFERENCE_MINS}m ${INFERENCE_SECS}s (${INFERENCE_DURATION}s)
Evaluation Time: ${EVALUATION_HOURS}h ${EVALUATION_MINS}m ${EVALUATION_SECS}s (${EVALUATION_DURATION}s)
Total Pipeline:  ${PIPELINE_HOURS}h ${PIPELINE_MINS}m ${PIPELINE_SECS}s (${PIPELINE_DURATION}s)

Breakdown:
----------
- Inference: ${INFERENCE_DURATION} seconds
- Evaluation: ${EVALUATION_DURATION} seconds
- Total: ${PIPELINE_DURATION} seconds

Comparison with OpenCompass (4 days = 345600 seconds):
-------------------------------------------------------
OpenCompass: 345600 seconds (4 days)
This Pipeline: ${PIPELINE_DURATION} seconds
Speedup: $(echo "scale=2; 345600 / ${PIPELINE_DURATION}" | bc)x faster
EOF

echo "============================================================"
echo "Pipeline Complete!"
echo "============================================================"
echo "Output Dir: $OUTPUT_DIR"
echo "Results: $RESULTS_DIR"
echo "Eval Logs: $LOGS_EVAL_DIR"
echo "Summary: $SUMMARY_DIR"
echo "Judge Logs: $JUDGE_LOG_DIR (extra, for resume)"
echo "Timing saved to: $TIMING_FILE"
echo ""

# Generate summary files (matching OC)
SUMMARY_BASE="$SUMMARY_DIR/summary_$(basename "$OUTPUT_DIR")"
echo "Generating summary files..."

# Create summary markdown
cat > "${SUMMARY_BASE}.md" <<EOF
# HealthBench Evaluation Summary

**Model:** $MODEL_NAME  
**Dev Subset:** $DEV_SUBSET  
**Concurrency:** $CONCURRENCY  
**Timestamp:** $(basename "$OUTPUT_DIR")

## Timing

- **Inference Time:** ${INFERENCE_HOURS}h ${INFERENCE_MINS}m ${INFERENCE_SECS}s (${INFERENCE_DURATION}s)
- **Evaluation Time:** ${EVALUATION_HOURS}h ${EVALUATION_MINS}m ${EVALUATION_SECS}s (${EVALUATION_DURATION}s)
- **Total Pipeline:** ${PIPELINE_HOURS}h ${PIPELINE_MINS}m ${PIPELINE_SECS}s (${PIPELINE_DURATION}s)

## Results

EOF

# Create summary CSV
echo "dataset,accuracy,accuracy_std,n_samples" > "${SUMMARY_BASE}.csv"
echo "timing,inference_seconds,evaluation_seconds,total_seconds" >> "${SUMMARY_BASE}.csv"
echo "timing,${INFERENCE_DURATION},${EVALUATION_DURATION},${PIPELINE_DURATION}" >> "${SUMMARY_BASE}.csv"

# Create summary text
cat > "${SUMMARY_BASE}.txt" <<EOF
HealthBench Evaluation Summary
===============================
Model: $MODEL_NAME
Dev Subset: $DEV_SUBSET
Concurrency: $CONCURRENCY
Timestamp: $(basename "$OUTPUT_DIR")

Timing:
-------
Inference Time:  ${INFERENCE_HOURS}h ${INFERENCE_MINS}m ${INFERENCE_SECS}s (${INFERENCE_DURATION}s)
Evaluation Time: ${EVALUATION_HOURS}h ${EVALUATION_MINS}m ${EVALUATION_SECS}s (${EVALUATION_DURATION}s)
Total Pipeline:  ${PIPELINE_HOURS}h ${PIPELINE_MINS}m ${PIPELINE_SECS}s (${PIPELINE_DURATION}s)

Results:
--------
EOF

# Collect results and write to summary
for SUBSET_NAME in base hard consensus; do
    case "$SUBSET_NAME" in
        base)
            OC_FILENAME="healthbench.json"
            ;;
        hard)
            OC_FILENAME="healthbench_hard.json"
            ;;
        consensus)
            OC_FILENAME="healthbench_consensus.json"
            ;;
    esac
    RESULTS_FILE="$RESULTS_DIR/$OC_FILENAME"
    if [ -f "$RESULTS_FILE" ]; then
        python3 <<PYTHON_SCRIPT
import json
import sys

results_file = '$RESULTS_FILE'
subset_name = '$SUBSET_NAME'
oc_filename = '$OC_FILENAME'
summary_base = '${SUMMARY_BASE}'

try:
    with open(results_file) as f:
        r = json.load(f)
    acc = r.get('accuracy')
    std = r.get('accuracy_std')
    n = r.get('n_samples', 0)
    
    if acc is not None:
        print(f'  {subset_name} ({oc_filename}):')
        print(f'    Accuracy: {acc:.4f} ± {std:.4f} (n={n})')
        
        # Write to CSV
        with open(f'{summary_base}.csv', 'a') as csv:
            csv.write(f'{oc_filename},{acc:.6f},{std:.6f},{n}\\n')
        
        # Write to markdown
        with open(f'{summary_base}.md', 'a') as md:
            md.write(f'### {oc_filename}\\n')
            md.write(f'- Accuracy: {acc:.4f} ± {std:.4f}\\n')
            md.write(f'- N Samples: {n}\\n\\n')
        
        # Write to text
        with open(f'{summary_base}.txt', 'a') as txt:
            txt.write(f'{oc_filename}: {acc:.4f} ± {std:.4f} (n={n})\\n')
    else:
        print('    No results')
except Exception as e:
    print(f'    (could not read: {e})', file=sys.stderr)
PYTHON_SCRIPT
    fi
done

echo ""
echo "Summary files created:"
echo "  - ${SUMMARY_BASE}.md"
echo "  - ${SUMMARY_BASE}.csv"
echo "  - ${SUMMARY_BASE}.txt"
