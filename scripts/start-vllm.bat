@echo off
REM Windows launcher for vLLM server in WSL2

echo ========================================
echo Starting HunyuanOCR vLLM Server
echo ========================================
echo.
echo This will start the vLLM server in WSL2
echo Server will be available at http://localhost:8000
echo Press Ctrl+C to stop
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR && ./scripts/start-vllm.sh"
