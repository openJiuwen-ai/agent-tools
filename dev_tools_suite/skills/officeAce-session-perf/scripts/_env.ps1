# Bundled ocperf + skill config loader.
function Resolve-OcperfHome {
    param([string]$ScriptsDir)
    $skillDir = Split-Path $ScriptsDir -Parent
    $bundled = Join-Path $skillDir "ocperf"
    if (Test-Path (Join-Path $bundled "src\ocperf\cli.py")) {
        return (Resolve-Path $bundled).Path
    }
    if ($env:OCPERF_HOME -and (Test-Path (Join-Path $env:OCPERF_HOME "src\ocperf\cli.py"))) {
        return (Resolve-Path $env:OCPERF_HOME).Path
    }
    throw 'ocperf engine not found (expected skill/ocperf).'
}

function Initialize-OcperfEnv {
    param([string]$ScriptsDir)
    $ocRoot = Resolve-OcperfHome -ScriptsDir $ScriptsDir
    $env:OCPERF_HOME = $ocRoot
    $env:PYTHONPATH = Join-Path $ocRoot 'src'
    return $ocRoot
}

function Import-OfficeClawSkillConfig {
    param([string]$ScriptsDir)
    $configDir = Join-Path (Split-Path $ScriptsDir -Parent) 'config'
    $localFile = Join-Path $configDir 'config.local.ps1'
    $cfg = @{ SessionsRoot = $env:OFFICECLAW_SESSIONS; LogsRoot = $env:OFFICECLAW_LOGS }
    if (Test-Path $localFile) {
        . $localFile
        if ($Script:OfficeClawSkillConfig) {
            foreach ($key in @('SessionsRoot', 'LogsRoot')) {
                $val = $Script:OfficeClawSkillConfig[$key]
                if ($val -and -not [string]::IsNullOrWhiteSpace([string]$val)) {
                    $cfg[$key] = [string]$val
                }
            }
        }
    }
    if ($cfg.SessionsRoot) { $env:OFFICECLAW_SESSIONS = $cfg.SessionsRoot }
    if ($cfg.LogsRoot) { $env:OFFICECLAW_LOGS = $cfg.LogsRoot }
    return $cfg
}

function Resolve-OfficeClawPaths {
    param([hashtable]$Config, [string]$SessionsRoot = '', [string]$LogsRoot = '')
    $sessions = if ($SessionsRoot) { $SessionsRoot } else { $Config.SessionsRoot }
    $logs = if ($LogsRoot) { $LogsRoot } else { $Config.LogsRoot }
    if (-not $sessions -or -not $logs) {
        throw 'Set config/config.local.ps1 (SessionsRoot, LogsRoot) or pass -SessionsRoot / -LogsRoot.'
    }
    return @{ SessionsRoot = $sessions; LogsRoot = $logs }
}
