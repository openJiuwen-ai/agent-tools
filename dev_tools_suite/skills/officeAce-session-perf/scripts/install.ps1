# Install this skill to ~/.cursor/skills/ (self-contained; includes bundled ocperf).
$ErrorActionPreference = "Stop"
$scriptsDir = $PSScriptRoot
$skillDir = Split-Path $scriptsDir -Parent

. (Join-Path $scriptsDir "_env.ps1")
Initialize-OcperfEnv -ScriptsDir $scriptsDir | Out-Null

$dest = Join-Path $env:USERPROFILE ".cursor\skills\officeclaw-session-perf"
Write-Host "Install -> $dest" -ForegroundColor Cyan
if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
Copy-Item -Recurse -Force $skillDir $dest
Write-Host "Done. Cursor loads: $dest\SKILL.md" -ForegroundColor Green
Write-Host "Copy config\config.example.ps1 to config\config.local.ps1 and set SessionsRoot / LogsRoot." -ForegroundColor Yellow
Write-Host "Restart Cursor." -ForegroundColor Yellow
