@echo off
REM Windows launcher for vLLM API tests in WSL2

echo ========================================
echo Testing HunyuanOCR vLLM API
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR && ./scripts/test-vllm.sh"

echo.
pause
