@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
start "Euphonia" ".venv\Scripts\pythonw.exe" "main.py"
