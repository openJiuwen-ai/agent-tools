# OfficeClaw session performance analysis — self-contained skill pipeline.
param(
    [Parameter(Mandatory = $true)]
    [string]$SessionId,
    [string]$OutDir = "",
    [string]$DataDir = "",
    [string]$LogsRoot = "",
    [string]$SessionsRoot = "",
    [switch]$Fusion
)
$ErrorActionPreference = "Stop"
$scriptsDir = $PSScriptRoot
. (Join-Path $scriptsDir "_env.ps1")
Initialize-OcperfEnv -ScriptsDir $scriptsDir | Out-Null
$skillConfig = Import-OfficeClawSkillConfig -ScriptsDir $scriptsDir
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$Session = $SessionId.Trim()
if (-not $Session.StartsWith("officeclaw_")) { $Session = "officeclaw_$Session" }
$safe = ($Session -replace '[^a-zA-Z0-9_-]', '_')
$resolvedOut = $null

if ($DataDir) {
    $resolvedOut = if ($OutDir) { $OutDir } else { Join-Path $DataDir "perf_reports" }
    New-Item -ItemType Directory -Force -Path $resolvedOut | Out-Null
    Write-Host "[1/4] OCPerf reports (history + full when present; fusion off by default)" -ForegroundColor Cyan
    $ocArgs = @("skill", $DataDir, "-s", $Session, "--out-dir", $resolvedOut)
    if ($Fusion) { $ocArgs += "--fusion" }
    & $python -m ocperf @ocArgs
    $code = $LASTEXITCODE
} else {
    $paths = Resolve-OfficeClawPaths -Config $skillConfig -SessionsRoot $SessionsRoot -LogsRoot $LogsRoot
    $ocArgs = @("skill", "--officeclaw", "-s", $SessionId)
    if ($OutDir) {
        $ocArgs += @("--out-dir", $OutDir)
        $resolvedOut = $OutDir
    }
    $ocArgs += @("--logs-root", $paths.LogsRoot, "--sessions-root", $paths.SessionsRoot)
    if ($Fusion) { $ocArgs += "--fusion" }
    Write-Host "[1/4] OCPerf OfficeClaw mode" -ForegroundColor Cyan
    Write-Host "  sessions: $($paths.SessionsRoot)"
    Write-Host "  logs:     $($paths.LogsRoot)"
    & $python -m ocperf @ocArgs
    $code = $LASTEXITCODE
    if (-not $resolvedOut) {
        $resolvedOut = Join-Path $paths.SessionsRoot "$Session\perf_reports"
    }
}
if ($code -ne 0) { exit $code }

$bundle = Join-Path $resolvedOut "skill_bundle_$safe.json"
$histHtml = Join-Path $resolvedOut "out_history_$safe.html"
if (-not (Test-Path $bundle)) { Write-Error "Missing bundle: $bundle" }
if (-not (Test-Path $histHtml)) { Write-Error "Missing required history HTML: $histHtml" }

$llmLatHtml = Join-Path $resolvedOut "out_llm_latency_$safe.html"
if (-not (Test-Path $llmLatHtml)) { Write-Error "Missing required LLM latency HTML: $llmLatHtml" }
Write-Host "OK: LLM latency timeline HTML" -ForegroundColor Green

$unifiedHtml = Join-Path $resolvedOut "out_unified_$safe.html"
if (-not (Test-Path $unifiedHtml)) { Write-Error "Missing required unified HTML: $unifiedHtml" }
Write-Host "OK: unified master HTML" -ForegroundColor Green

$fullHtml = Join-Path $resolvedOut "out_full_$safe.html"
if (Test-Path $fullHtml) {
    Write-Host "OK: history + full HTML" -ForegroundColor Green
} else {
    Write-Host "OK: history HTML (no full report)" -ForegroundColor Green
}

Write-Host "[2/4] E2E flowchart (HTML + Mermaid + SVG)" -ForegroundColor Cyan
& $python (Join-Path $scriptsDir "generate_flowchart_svg.py") --bundle $bundle --out-dir $resolvedOut
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$flowHtml = Join-Path $resolvedOut "execution_flowchart_$safe.html"
$flowSvg = Join-Path $resolvedOut "execution_flowchart_$safe.svg"
$mermaidVal = Join-Path $resolvedOut "execution_flow_mermaid_${safe}_validation.json"
if (-not (Test-Path $flowHtml)) { Write-Error "Missing flowchart HTML: $flowHtml" }
if (-not (Test-Path $flowSvg)) { Write-Error "Missing flowchart SVG: $flowSvg" }
if (-not (Test-Path $mermaidVal)) { Write-Error "Missing Mermaid validation: $mermaidVal" }
$valCheck = & $python -c "import json,sys; d=json.load(open(sys.argv[1],encoding='utf-8')); print('OK' if d.get('ok') else 'FAIL'); sys.exit(0 if d.get('ok') else 1)" $mermaidVal
if ($LASTEXITCODE -ne 0) { Write-Error "Mermaid validation failed (see $mermaidVal)" }
Write-Host "Mermaid validation: $valCheck" -ForegroundColor Green

Write-Host "[3/4] Analysis Markdown" -ForegroundColor Cyan
& $python (Join-Path $scriptsDir "generate_analysis_md.py") --bundle $bundle --out (Join-Path $resolvedOut "analysis_report.md")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/4] Done -> $resolvedOut" -ForegroundColor Cyan
Get-ChildItem $resolvedOut -File | ForEach-Object { Write-Host "  $($_.Name)" }
Write-Host ""
Write-Host "Next: open execution_flowchart_$safe.html; complete analysis_report.md required sections" -ForegroundColor Yellow
exit 0
