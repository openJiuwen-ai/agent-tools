# OCPerf CLI wrapper (uses skill/ocperf).
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$OcperfArgs
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_env.ps1")
Initialize-OcperfEnv -ScriptsDir $PSScriptRoot | Out-Null
$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $python -m ocperf @OcperfArgs
exit $LASTEXITCODE
