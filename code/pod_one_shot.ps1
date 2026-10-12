# Run:  cd T5_revision\code ; .\pod_one_shot.ps1
$ErrorActionPreference = "Stop"
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"
$scp = "C:\Windows\System32\OpenSSH\scp.exe"
$key = "$env:USERPROFILE\.ssh\id_ed25519"
$hostId = "m1zsr66lo1pavx-644120b8@ssh.runpod.io"
$proj = Split-Path $PSScriptRoot -Parent
$code = $PSScriptRoot
$parquet = Join-Path $proj "data\processed\features.parquet"
if (-not (Test-Path $parquet)) {
    $parquet = Join-Path $code "data\processed\features.parquet"
}
$bootstrap = Join-Path $code "_pod_remote_bootstrap.sh"

Write-Host "=== Step 1: remote setup (clone + pip) ==="
Get-Content $bootstrap -Raw | & $ssh -t -i $key $hostId "bash -s"
if ($LASTEXITCODE -ne 0) { throw "Remote setup failed exit $LASTEXITCODE" }

Write-Host "=== Step 2: upload parquet ==="
if (Test-Path $parquet) {
    & $scp -i $key $parquet "${hostId}:/workspace/ML-Model/code/data/processed/features.parquet"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "SCP parquet failed. Use RunPod upload to code/data/processed/features.parquet"
    }
} else {
    Write-Warning "No parquet at $parquet"
}

Write-Host "=== Step 3: upload outputs ==="
$out = Join-Path $code "outputs"
if (Test-Path $out) {
    & $scp -r -i $key $out "${hostId}:/workspace/ML-Model/code/"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "SCP outputs failed. Use RunPod upload to code/outputs/"
    }
}

Write-Host "=== Step 4: eval ==="
$evalSh = Join-Path $code "_pod_remote_eval.sh"
Get-Content $evalSh -Raw | & $ssh -t -i $key $hostId "bash -s"
Write-Host "Done. Get zip from /workspace/ML-Model/deliverables/T5_milestone_compute.zip"
