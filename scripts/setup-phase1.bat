@echo off
REM Windows launcher for Phase 1 setup in WSL2

echo ========================================
echo HunyuanOCR Phase 1 Setup (WSL2)
echo ========================================
echo.

echo Checking WSL2...
wsl --list --verbose

echo.
echo Starting setup in WSL2...
echo This will install all dependencies in Ubuntu WSL2
echo.

wsl bash -c "cd /mnt/c/Users/akona/OneDrive/Dev/HunyuanOCR && chmod +x scripts/*.sh && ./scripts/setup-phase1.sh"

echo.
echo Setup complete!
echo.
pause
