param([int]$Workers = 2)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.12 with the Windows py launcher, then retry.' }
& py -3.12 (Join-Path $PSScriptRoot 'download_package.py') --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Download or verification did not finish. Read the error above; a rerun can resume downloaded parts.' }
