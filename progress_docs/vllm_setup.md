# Deploy Qwen3-4B via vLLM

## Goal

Deploy local vLLM server for Qwen3-4B-Thinking-2507 to enable high-throughput inference.

## Commands

### Step 1: Check GPU availability

```bash
nvidia-smi
```

### Step 2: Install vLLM (in opencompass env)

```bash
conda activate opencompass
pip install vllm
```

### Step 3: Launch vLLM Server

```bash
vllm serve /data1/models/Qwen3-4B-Thinking-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --max-model-len 16384 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1
```

Key flags:

- `--gpu-memory-utilization 0.9`: Use 90% of GPU memory
- `--max-model-len 16384`: Match config's max_seq_len
- `--enable-reasoning --reasoning-parser deepseek_r1`: Parse thinking tokens (Qwen3-Thinking uses `<think>` tags like DeepSeek-R1)

### Step 4: Test the deployment

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data1/models/Qwen3-4B-Thinking-2507",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'
```

# Test gpt-oss-120b (use served-model-name, not full path)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100
  }'

## After Deployment

Update `task3_eval.py`:

- Change student model `openai_api_base` to `http://localhost:8000/v1`
- Increase `num_worker` to 50-100
- Increase `batch_size` to 64+