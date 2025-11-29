#!/bin/bash
# Phase 1 Setup Script for WSL2
# This script sets up the HunyuanOCR environment in WSL2

set -e

echo "========================================"
echo "HunyuanOCR Phase 1 Setup"
echo "========================================"
echo ""

# Get project directory
PROJECT_DIR="/mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR"
cd "$PROJECT_DIR"

echo "Step 1: Checking Python version and dependencies..."
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found."
    echo "Please run this command in WSL2:"
    echo "  sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip"
    exit 1
else
    # Python exists, but check if venv package is installed
    if ! dpkg -l | grep -q python3.12-venv; then
        echo "python3.12-venv package is required."
        echo ""
        echo "Please enter your WSL2 password to install it..."
        echo "(This is the password you use for Ubuntu in WSL2)"
        echo ""
        if ! sudo apt update || ! sudo apt install -y python3.12-venv; then
            echo ""
            echo "ERROR: Could not install python3.12-venv"
            echo "Please run manually: sudo apt install -y python3.12-venv"
            exit 1
        fi
    fi
fi

python3.12 --version
echo "✓ Python 3.12 and venv available"
echo ""

echo "Step 2: Creating virtual environment..."
VENV_PATH="$PROJECT_DIR/hunyuanocr-env"
# Remove incomplete venv if activate script is missing
if [ -d "$VENV_PATH" ] && [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Removing incomplete virtual environment..."
    rm -rf "$VENV_PATH"
fi

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment (this may take a minute)..."
    python3.12 -m venv "$VENV_PATH" --copies
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

echo "Step 3: Verifying and activating environment..."
VENV_PATH="$PROJECT_DIR/hunyuanocr-env"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: Virtual environment is corrupted!"
    echo "This can happen with OneDrive sync. Recreating with --copies flag..."
    rm -rf "$VENV_PATH"
    python3.12 -m venv "$VENV_PATH" --copies
fi

# Use direct python path instead of activate for more reliability
PYTHON_BIN="$VENV_PATH/bin/python"
echo "Using Python: $PYTHON_BIN"
$PYTHON_BIN -m pip install --upgrade pip
echo "✓ Pip upgraded"
echo ""

echo "Step 4: Installing vLLM (this may take several minutes)..."
$PYTHON_BIN -m pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
echo "✓ vLLM installed"
echo ""

echo "Step 5: Installing project dependencies..."
$PYTHON_BIN -m pip install -r "$PROJECT_DIR/requirements.txt"
echo "✓ Project dependencies installed"
echo ""

echo "Step 6: Installing transformers (specific commit)..."
$PYTHON_BIN -m pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
echo "✓ Transformers installed"
echo ""

echo "Step 7: Installing additional dependencies..."
$PYTHON_BIN -m pip install openai tqdm pillow aiohttp python-dotenv
echo "✓ Additional dependencies installed"
echo ""

echo "Step 8: Installing MCP server dependencies..."
$PYTHON_BIN -m pip install -r "$PROJECT_DIR/docs/mcp-server/requirements.txt"
echo "✓ MCP dependencies installed"
echo ""

echo "========================================"
echo "✓ Phase 1 Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test with Transformers: cd Hunyuan-OCR-master/Hunyuan-OCR-hf && python run_hy_ocr.py"
echo "2. Start vLLM server: ./scripts/start-vllm.sh"
echo "3. Test vLLM API: cd docs/mcp-server && python test_vllm.py"
echo ""
