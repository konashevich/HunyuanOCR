#!/bin/bash
# Test vLLM API functionality

PROJECT_DIR="/mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR"
cd "$PROJECT_DIR"

echo "========================================"
echo "Testing HunyuanOCR vLLM API"
echo "========================================"
echo ""

# Activate environment
source "$PROJECT_DIR/hunyuanocr-env/bin/activate"

# Navigate to MCP server directory
cd "$PROJECT_DIR/docs/mcp-server"

# Run tests
python test_vllm.py
