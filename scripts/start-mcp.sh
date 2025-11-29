#!/bin/bash
# Start MCP Server

PROJECT_DIR="/mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR"
cd "$PROJECT_DIR"

echo "========================================"
echo "Starting HunyuanOCR MCP Server"
echo "========================================"
echo ""
echo "Make sure vLLM server is running first!"
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Activate environment
source "$PROJECT_DIR/hunyuanocr-env/bin/activate"

# Navigate to MCP server
cd "$PROJECT_DIR/docs/mcp-server"

# Start MCP server
python server.py
