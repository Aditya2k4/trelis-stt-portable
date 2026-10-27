param([switch]$Training)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$environmentName = if ($Training) { '.venv-training' } else { '.venv-asr' }
$pythonPath = Join-Path $PSScriptRoot "$environmentName\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & py -3.12 -m venv $environmentName
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 with its Windows py launcher, then retry.' }
}
& $pythonPath -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
if ($Training) {
    & $pythonPath -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/xpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch XPU installation failed.' }
    & $pythonPath -m pip install -r fine-tuning/requirements.txt
} else {
    & $pythonPath -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch CPU installation failed.' }
    & $pythonPath -m pip install -r requirements-inference.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $pythonPath -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts found; inspect before proceeding.' }
Write-Host "Environment ready: $pythonPath"
if (-not $Training) { & $pythonPath diagnose.py }
