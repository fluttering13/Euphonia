@echo off
cd /d "%~dp0"
where uv >nul 2>nul
if errorlevel 1 (
  echo Please install uv first: python -m pip install uv
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" uv venv .venv --python 3.12
if errorlevel 1 goto fail
.venv\Scripts\python.exe -c "import torch, torchaudio; assert torch.cuda.is_available()" >nul 2>nul
if not errorlevel 1 goto deps
uv pip install --python .venv\Scripts\python.exe torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail
:deps
.venv\Scripts\python.exe -m pip --version >nul 2>nul
if not errorlevel 1 (
  .venv\Scripts\python.exe -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  goto done
)
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
if errorlevel 1 goto fail
:done
echo Setup complete. Open start.bat to launch Euphonia.
pause
exit /b 0
:fail
echo Setup failed. See the error above and retry.
pause
exit /b 1
