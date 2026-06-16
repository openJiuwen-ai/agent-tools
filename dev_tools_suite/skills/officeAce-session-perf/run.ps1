# Entry point: analyze one OfficeClaw session by ID (paths from config/config.local.ps1).
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
$skillScript = Join-Path $PSScriptRoot "scripts\run_officeclaw_analysis.ps1"
$params = @{
    SessionId    = $SessionId
    OutDir       = $OutDir
    DataDir      = $DataDir
    LogsRoot     = $LogsRoot
    SessionsRoot = $SessionsRoot
}
if ($Fusion) { $params.Fusion = $true }
& $skillScript @params
exit $LASTEXITCODE
