# Live crane demo on Windows (PowerShell). Usage, from the repo folder:
#   powershell -ExecutionPolicy Bypass -File crane\run_live.ps1            # live pick and place, TabPFN chooses the anti-sway
#   powershell -ExecutionPolicy Bypass -File crane\run_live.ps1 predict    # live forecast only
#   powershell -ExecutionPolicy Bypass -File crane\run_live.ps1 pick --stopped --manual
# Needs: uv (https://docs.astral.sh/uv/getting-started/installation/), internet on first run, .env with TABPFN_TOKEN.
param([string]$App = "pick")
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Host "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1 }
if (-not (Test-Path "crane\.venv")) { uv venv --python 3.12 crane\.venv }
uv pip install --python crane\.venv\Scripts\python.exe -q -r crane\requirements.txt
if (Test-Path ".env") {
  Get-Content ".env" | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2].Trim().Trim('"').Trim("'")) }
  }
}
if (-not $env:TABPFN_TOKEN) { Write-Host "Put TABPFN_TOKEN=<your key> in .env (platform.priorlabs.ai, accept the TabPFN-3.5 licence)"; exit 1 }
$env:TABPFN_NO_BROWSER = "1"
$py = "crane\.venv\Scripts\python.exe"
if ($App -eq "predict") { & $py crane\live_crane.py @args } else { & $py crane\live_pick_place.py --auto-tune @args }
