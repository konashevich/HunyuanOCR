@echo off
REM Windows launcher for MCP server in WSL2

echo ========================================
echo Starting HunyuanOCR MCP Server
echo ========================================
echo.
echo Make sure vLLM server is running first!
echo Press Ctrl+C to stop
echo ========================================
echo.

wsl bash -c "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR && ./scripts/start-mcp.sh"
