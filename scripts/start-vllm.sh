#!/bin/bash
# Start vLLM Server with optimized settings for RTX 3060 (6GB VRAM)

PROJECT_DIR="/mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR"
cd "$PROJECT_DIR"

echo "========================================"
echo "Starting HunyuanOCR vLLM Server"
echo "========================================"
echo ""
echo "GPU: RTX 3060 (6GB VRAM)"
echo "Optimization: Reduced memory utilization"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Activate environment
source "$PROJECT_DIR/hunyuanocr-env/bin/activate"

# Start vLLM with ultra-minimal memory settings for 6GB GPU
vllm serve tencent/HunyuanOCR \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --gpu-memory-utilization 0.05 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype half \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --enforce-eager

# Alternative CPU mode (uncomment if GPU OOM):
# vllm serve tencent/HunyuanOCR \
#     --device cpu \
#     --host 0.0.0.0 \
#     --port 8000
