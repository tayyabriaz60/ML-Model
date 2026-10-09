# Fourth deliverable: code + key outputs (exclude 384MB features if already off-box).
$proj = Split-Path $PSScriptRoot -Parent
$code = $PSScriptRoot
$out = Join-Path $proj "deliverables\T5_code_outputs.zip"
$staging = Join-Path $env:TEMP "t5_code_staging"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging | Out-Null
Copy-Item $code\*.py $staging\
Copy-Item $code\src $staging\src -Recurse
Copy-Item $code\requirements.txt $staging\
Copy-Item $code\outputs\tables $staging\outputs\tables -Recurse
Copy-Item $code\outputs\figures $staging\outputs\figures -Recurse
Copy-Item $code\outputs\xai $staging\outputs\xai -Recurse
Copy-Item $code\outputs\manuscript_numbers.json $staging\outputs\
if (Test-Path $out) { Remove-Item $out }
Compress-Archive -Path "$staging\*" -DestinationPath $out
Write-Host "wrote $out"
