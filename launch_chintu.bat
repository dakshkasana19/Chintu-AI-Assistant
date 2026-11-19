@echo off
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Virtual environment not found. Run run_chintu.bat first to set up Chintu.
    exit /b 1
)

start "" "venv\Scripts\python.exe" "chintu_assistant.py"
