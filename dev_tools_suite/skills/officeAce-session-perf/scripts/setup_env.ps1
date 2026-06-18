# Validate self-contained officeclaw-session-perf environment.
$ErrorActionPreference = "Stop"
$scriptsDir = $PSScriptRoot
. (Join-Path $scriptsDir "_env.ps1")
$ocperfHome = Initialize-OcperfEnv -ScriptsDir $scriptsDir
$config = Import-OfficeClawSkillConfig -ScriptsDir $scriptsDir
$skillDir = Split-Path $scriptsDir -Parent

Write-Host "=== officeclaw-session-perf env check ===" -ForegroundColor Cyan
Write-Host "Skill: $skillDir"
Write-Host "OCPERF_HOME: $ocperfHome"

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ver = & $python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Host "Python: $ver" -ForegroundColor Green
& $python -m ocperf skill --help | Out-Null
Write-Host "ocperf CLI: OK" -ForegroundColor Green

Write-Host ""
Write-Host "Configured paths:" -ForegroundColor Cyan
foreach ($key in @("SessionsRoot", "LogsRoot")) {
    $p = $config[$key]
    if ($p -and (Test-Path $p)) {
        Write-Host "  $key : $p" -ForegroundColor Green
    } elseif ($p) {
        Write-Host "  $key : $p (path not found)" -ForegroundColor Yellow
    } else {
        Write-Host "  $key : not set — copy config\config.example.ps1 to config\config.local.ps1" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Run: .\run.ps1 -SessionId <session_id>" -ForegroundColor Cyan
