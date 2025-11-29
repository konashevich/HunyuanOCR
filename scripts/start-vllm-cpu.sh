#!/bin/bash
# Start vLLM Server in CPU mode (fallback for memory constraints)

PROJECT_DIR="/mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR"
cd "$PROJECT_DIR"

echo "========================================"
echo "Starting HunyuanOCR vLLM Server (CPU Mode)"
echo "========================================"
echo ""
echo "Note: CPU mode is slower but works with limited GPU memory"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Activate environment
source "$PROJECT_DIR/hunyuanocr-env/bin/activate"

# Start vLLM in CPU mode
vllm serve tencent/HunyuanOCR \
    --device cpu \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096
