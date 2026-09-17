param([ValidateSet('all', 'f5', 'zipvoice')][string]$Model = 'all')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Run-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) { throw 'GPU model environment setup failed.' }
}

$gpuPackages = @('torch==2.7.1+cu128', 'torchaudio==2.7.1+cu128', '--extra-index-url', 'https://download.pytorch.org/whl/cu128', '--index-strategy', 'unsafe-best-match')
foreach ($backend in @('f5', 'zipvoice')) {
    if ($Model -ne 'all' -and $Model -ne $backend) { continue }
    $pythonPath = Join-Path $PSScriptRoot ".venv-$backend/Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        Run-Uv venv ".venv-$backend" --python '.venv/Scripts/python.exe'
    }
    $torchWheel = Join-Path $PSScriptRoot 'data/wheels/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
    if (Test-Path -LiteralPath $torchWheel) {
        Run-Uv pip install --python $pythonPath --no-deps $torchWheel
    }

    $repoName = if ($backend -eq 'f5') { 'F5-TTS' } else { 'ZipVoice' }
    $repoUrl = if ($backend -eq 'f5') { 'https://github.com/SWivid/F5-TTS.git' } else { 'https://github.com/k2-fsa/ZipVoice.git' }
    $revision = if ($backend -eq 'f5') { '9c614e9657089213efc6a7421b30630be138a3f5' } else { '2f7326fbfe999a3ad179e3f1af82a424d4a62819' }
    $sourcePath = Join-Path $PSScriptRoot "experiments/$repoName"
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        & git clone $repoUrl $sourcePath
        if ($LASTEXITCODE -ne 0) { throw 'Source download failed.' }
        & git -C $sourcePath checkout $revision
        if ($LASTEXITCODE -ne 0) { throw 'Source revision checkout failed.' }
    }
    if ($backend -eq 'f5') {
        Run-Uv pip install --python $pythonPath --no-deps -e $sourcePath
        Run-Uv pip install --python $pythonPath @gpuPackages 'numpy==1.26.4' 'transformers==4.57.3' 'librosa==0.11.0' cached_path hydra-core ema-pytorch torchdiffeq vocos x-transformers pypinyin rjieba soundfile matplotlib accelerate wandb pydub datasets
    } else {
        Run-Uv pip install --python $pythonPath @gpuPackages 'numpy==1.26.4' 'librosa==0.11.0' requests -r "$sourcePath/requirements.txt"
    }
    & '.venv/Scripts/python.exe' scripts/download_models.py $backend
    if ($LASTEXITCODE -ne 0) { throw "$backend weights are not ready." }
    Write-Host "$backend environment and weights prepared."
}
