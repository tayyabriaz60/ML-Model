# Four outputs for Jarwal milestone (compute only)
$proj = Split-Path $PSScriptRoot -Parent
$code = $PSScriptRoot
$out = Join-Path $proj "deliverables\T5_milestone_compute.zip"
$staging = Join-Path $env:TEMP "t5_milestone_staging"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item "$code\outputs\tables\bootstrap_ci.csv" $staging\
Copy-Item "$code\outputs\tables\per_station_metrics.csv" $staging\
Copy-Item "$code\outputs\tables\table09_regime_shap.csv" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\ci_mask_report.txt" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\station_coverage_note.txt" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\station_coverage_main_h136.csv" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\tree_preflight_report.txt" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\h1_xgb_lgb_ci_overlap.txt" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\tables\run_env.txt" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\figures\F8_regime_shap.png" $staging -ErrorAction SilentlyContinue
Copy-Item "$code\outputs\figures\F8_regime_shap.pdf" $staging -ErrorAction SilentlyContinue
if (Test-Path $out) { Remove-Item $out }
Compress-Archive -Path "$staging\*" -DestinationPath $out
Write-Host "wrote $out"
