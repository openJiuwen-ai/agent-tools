# Remove cache and dev-only files before uploading the skill package.
$ErrorActionPreference = "Stop"
$skillDir = Split-Path $PSScriptRoot -Parent
$removed = 0
foreach ($pattern in @("__pycache__", ".pytest_cache")) {
    Get-ChildItem $skillDir -Recurse -Directory -Force -Filter $pattern -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-Item $_.FullName -Recurse -Force; $removed++ }
}
Get-ChildItem $skillDir -Recurse -File -Force -Filter "*.pyc" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Force; $removed++ }
Write-Host "Cleaned $removed cache entries under $skillDir" -ForegroundColor Green
$count = (Get-ChildItem $skillDir -Recurse -File -Force).Count
Write-Host "Remaining files: $count" -ForegroundColor Cyan
