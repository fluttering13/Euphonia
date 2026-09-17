param([ValidateSet('all', 'faster', 'qwen_streaming', 'f5', 'zipvoice')][string]$Model = 'all')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Run-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw 'Model environment setup failed.' }
}

$basePython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'

if ($Model -in @('all', 'faster', 'qwen_streaming')) {
    $envPath = Join-Path $PSScriptRoot '.venv-faster'
    $pythonPath = Join-Path $envPath 'Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        Run-Uv venv $envPath --python $basePython
    }
    $repoPath = Join-Path $PSScriptRoot 'experiments/faster-qwen3-tts'
    if (-not (Test-Path -LiteralPath $repoPath)) {
        & git clone --depth 1 https://github.com/andimarafioti/faster-qwen3-tts.git $repoPath
        if ($LASTEXITCODE -ne 0) { throw 'Faster Qwen3-TTS source download failed.' }
    }
    Push-Location $repoPath
    try {
        Run-Uv pip install --python $pythonPath -e . 'torch==2.5.1+cu124' 'torchaudio==2.5.1+cu124' --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
    } finally {
        Pop-Location
    }
    & $basePython scripts/download_models.py qwen
    if ($LASTEXITCODE -ne 0) { throw 'Qwen3-TTS model download failed.' }
}

if ($Model -in @('f5', 'zipvoice')) {
    & (Join-Path $PSScriptRoot 'setup_extra_models.ps1') -Model $Model
    exit $LASTEXITCODE
}

if ($Model -eq 'all') {
    & (Join-Path $PSScriptRoot 'setup_extra_models.ps1') -Model all
    if ($LASTEXITCODE -ne 0) { throw 'F5-TTS or ZipVoice setup failed.' }
}

Write-Host 'Model setup finished.'
