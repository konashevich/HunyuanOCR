@echo off
REM Windows launcher for vLLM server in CPU mode

echo ========================================
echo Starting HunyuanOCR vLLM Server (CPU)
echo ========================================
echo.
echo Server will be available at http://localhost:8000
echo CPU mode is slower but works with limited memory
echo Press Ctrl+C to stop
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR && ./scripts/start-vllm-cpu.sh"
