@echo off
setlocal enabledelayedexpansion

echo ========================================
echo   OfficeClaw 日志采集工具
echo ========================================
echo.

echo 请输入采集日志的时间范围
echo.

for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd'"') do set "TODAY=%%i"

:input_start
set "START_TIME="
set /p "START_TIME=请输入开始时间（例如：%TODAY% 00:00:00）: "
if "!START_TIME!"=="" (
    echo 错误：开始时间不能为空
    goto :input_start
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { try { [datetime]::Parse('%START_TIME%') } catch { Write-Host '错误：时间格式不正确，请使用格式：YYYY-MM-DD HH:MM:SS'; exit 1 } }"
if errorlevel 1 goto :input_start

:input_end
set "END_TIME="
set /p "END_TIME=请输入结束时间（例如：%TODAY% 23:59:59）: "
if "!END_TIME!"=="" (
    echo 错误：结束时间不能为空
    goto :input_end
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { try { [datetime]::Parse('%END_TIME%') } catch { Write-Host '错误：时间格式不正确，请使用格式：YYYY-MM-DD HH:MM:SS'; exit 1 } }"
if errorlevel 1 goto :input_end

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $d1=[datetime]::Parse('%START_TIME%'); $d2=[datetime]::Parse('%END_TIME%'); if ($d2 -lt $d1) { Write-Host '错误：结束时间不能早于开始时间'; exit 1 } }"
if errorlevel 1 goto :input_end

echo.
echo 时间范围：!START_TIME! ~ !END_TIME!
echo.
set "CONFIRM="
set /p "CONFIRM=确认开始采集吗(Y/N): "
if /i "!CONFIRM!" neq "Y" (
    echo 已取消操作
    pause
    exit /b 0
)

echo.
echo 正在查找OfficeClaw安装目录...
echo.

set "INSTALL_DIR="

for /f "skip=2 tokens=2*" %%a in ('reg query "HKCU\Software\ClowderLabs\OfficeClaw" /v "InstallDir" 2^>nul') do (
    set "INSTALL_DIR=%%b"
)

if not defined INSTALL_DIR (
    for /f "skip=2 tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OfficeClaw" /v "InstallLocation" 2^>nul') do (
        set "INSTALL_DIR=%%b"
    )
)

if not defined INSTALL_DIR (
    for /f "skip=2 tokens=2*" %%a in ('reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\OfficeClaw" /v "InstallLocation" 2^>nul') do (
        set "INSTALL_DIR=%%b"
    )
)

if not defined INSTALL_DIR (
    for /f "skip=2 tokens=2*" %%a in ('reg query "HKCU\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\OfficeClaw" /v "InstallLocation" 2^>nul') do (
        set "INSTALL_DIR=%%b"
    )
)

if not defined INSTALL_DIR (
    if exist "%LOCALAPPDATA%\Programs\OfficeClaw\OfficeClaw.exe" (
        set "INSTALL_DIR=%LOCALAPPDATA%\Programs\OfficeClaw"
    )
)

if not defined INSTALL_DIR (
    if exist "%ProgramFiles%\OfficeClaw\OfficeClaw.exe" (
        set "INSTALL_DIR=%ProgramFiles%\OfficeClaw"
    )
)

if not defined INSTALL_DIR (
    if exist "%ProgramFiles(x86)%\OfficeClaw\OfficeClaw.exe" (
        set "INSTALL_DIR=%ProgramFiles(x86)%\OfficeClaw"
    )
)

if not defined INSTALL_DIR (
    for /f "delims=" %%i in ('dir /b /s "%USERPROFILE%\Desktop\OfficeClaw.lnk" 2^>nul') do (
        for /f "skip=1 tokens=*" %%j in ('powershell -NoProfile -Command "(New-Object -ComObject WScript.Shell).CreateShortcut('%%i').TargetPath" 2^>nul') do (
            for %%k in ("%%j") do set "INSTALL_DIR=%%~dpk"
        )
    )
)

if defined INSTALL_DIR (
    if exist "%INSTALL_DIR%\OfficeClaw.exe" (
        echo 找到安装目录：%INSTALL_DIR%
        echo.
    ) else (
        set "INSTALL_DIR="
    )
)

if not defined INSTALL_DIR (
    echo 未能自动找到OfficeClaw安装目录，请手动输入：
    set /p "INSTALL_DIR=安装目录路径: "
    if not exist "%INSTALL_DIR%\OfficeClaw.exe" (
        echo 错误：指定的目录不是有效的OfficeClaw安装目录
        pause
        exit /b 1
    )
)

set "OUTPUT_DIR=%~dp0"

echo.
echo 正在采集...
echo.

for /f "tokens=1-3 delims=/ " %%a in ("%date%") do set "d=%%a%%b%%c"
set "now=%time%"
set "t=%now:~0,2%%now:~3,2%%now:~6,2%"
set "t=%t: =0%"
set "BUNDLE_NAME=office_claw_log_%d%_%t%"

set "BASE_DIR=%USERPROFILE%\.office-claw\.jiuwenclaw\service_default"
set "LOGS_DIR=%BASE_DIR%\.logs"
set "SESSIONS_DIR=%BASE_DIR%\agent_default\agent\sessions"

set "TEMP_DIR=%TEMP%\jiuwenclaw-%RANDOM%"
set "STAGE=%TEMP_DIR%\%BUNDLE_NAME%"

if not exist "%LOGS_DIR%" (
    echo 错误：运行时日志目录不存在：%LOGS_DIR%
    pause
    exit /b 1
)

echo 创建临时目录...
mkdir "%STAGE%\runtime_logs" 2>nul
mkdir "%STAGE%\install_logs" 2>nul
mkdir "%STAGE%\sessions" 2>nul

echo.
echo 采集运行时日志（含归档日志）...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $src='%LOGS_DIR%'; $dst='%STAGE%\runtime_logs'; Get-ChildItem -LiteralPath $src -Recurse -File | Where-Object { $_.Name -match '\.log(\.\d+)?$' } | ForEach-Object { $rel = $_.FullName; if ($rel.StartsWith($src)) { $rel = $rel.Substring($src.Length) }; if ($rel.StartsWith('\')) { $rel = $rel.Substring(1) }; $target = Join-Path $dst $rel; $dir = Split-Path $target -Parent; if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }; Copy-Item -LiteralPath $_.FullName -Destination $target -Force }; Write-Host ('已采集 ' + (Get-ChildItem -LiteralPath $dst -Recurse -File).Count + ' 个文件') }"

echo.
echo 过滤运行时日志...
set "OC_DST=%STAGE%\runtime_logs"
set "OC_START=!START_TIME!"
set "OC_END=!END_TIME!"
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $ErrorActionPreference='Stop'; $Rx=[regex]::new('^\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?)'); function LineTs([string]$ln) { if ([string]::IsNullOrWhiteSpace($ln)) { return $null }; $m=$Rx.Match($ln); if (-not $m.Success) { return $null }; try { return [datetime]::Parse($m.Groups[1].Value,[System.Globalization.CultureInfo]::InvariantCulture,[System.Globalization.DateTimeStyles]::None) } catch { return $null } }; function FirstTs([string]$p) { $sr=$null; try { $sr=New-Object System.IO.StreamReader($p,[System.Text.Encoding]::UTF8,$true); $n=0; while ($null -ne ($ln=$sr.ReadLine()) -and $n -lt 80) { $n++; $t=LineTs $ln; if ($null -ne $t) { return $t } } } finally { if ($null -ne $sr) { $sr.Close() } }; return $null }; function LastTs([string]$p) { $fs=$null; try { $fs=[System.IO.File]::OpenRead($p); $len=$fs.Length; if ($len -eq 0) { return $null }; $chunk=[Math]::Min([int64]8388608,$len); $fs.Position=$len-$chunk; $buf=New-Object byte[] $chunk; [void]$fs.Read($buf,0,$chunk); $txt=[System.Text.Encoding]::UTF8.GetString($buf); $arr=[regex]::Split($txt,'\r?\n'); for ($i=$arr.Length-1; $i -ge 0; $i--) { $t=LineTs $arr[$i]; if ($null -ne $t) { return $t } } } finally { if ($null -ne $fs) { $fs.Close() } }; return $null }; $dst=$env:OC_DST; $start=[datetime]::Parse($env:OC_START); $end=[datetime]::Parse($env:OC_END); Get-ChildItem -LiteralPath $dst -Recurse -File | ForEach-Object { $keep=$true; if ($_.Length -eq 0) { $keep=$false } else { $f=FirstTs $_.FullName; $g=LastTs $_.FullName; if ($null -eq $f -or $null -eq $g) { $keep=$true } elseif ($f -gt $end) { $keep=$false } elseif ($g -lt $start) { $keep=$false } }; if (-not $keep) { Remove-Item -LiteralPath $_.FullName -Force } } }"
if errorlevel 1 (
    echo ERROR: Runtime log filter failed.
    pause
    exit /b 1
)
set "OC_DST="
set "OC_START="
set "OC_END="

echo.
echo 采集安装目录日志...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $src='%INSTALL_DIR%\logs'; $dst='%STAGE%\install_logs'; $start=[datetime]::Parse('%START_TIME%'); if (Test-Path $src) { Get-ChildItem -LiteralPath $src -Filter '*.log' | Where-Object { $_.LastWriteTime -ge $start } | ForEach-Object { Copy-Item $_.FullName $dst -Force } }; Write-Host ('已采集 ' + ((Get-ChildItem -LiteralPath $dst -Filter '*.log' -ErrorAction SilentlyContinue).Count) + ' 个安装目录日志文件') }"

echo.
echo 采集审计日志...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $src='%INSTALL_DIR%\data\audit-logs'; $dst='%STAGE%\install_logs'; $start='%START_TIME%'.Substring(0,10); $end='%END_TIME%'.Substring(0,10); if (Test-Path $src) { Get-ChildItem -LiteralPath $src -Filter 'audit-*.ndjson' | Where-Object { $_.Name -match 'audit-(\d{4}-\d{2}-\d{2})\.ndjson' -and $Matches[1] -ge $start -and $Matches[1] -le $end } | ForEach-Object { Copy-Item $_.FullName $dst -Force } }; Write-Host ('已采集 ' + ((Get-ChildItem -LiteralPath $dst -Filter 'audit-*.ndjson' -ErrorAction SilentlyContinue).Count) + ' 个审计日志文件') }"

echo.
echo 采集API日志...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $src='%INSTALL_DIR%\data\logs\api'; $dst='%STAGE%\install_logs'; $start='%START_TIME%'.Substring(0,10); $end='%END_TIME%'.Substring(0,10); if (Test-Path $src) { Get-ChildItem -LiteralPath $src -Filter 'api.*.log' | Where-Object { $_.Name -match 'api\.(\d{4}-\d{2}-\d{2})\.' -and $Matches[1] -ge $start -and $Matches[1] -le $end } | ForEach-Object { Copy-Item $_.FullName $dst -Force } }; Write-Host ('已采集 ' + ((Get-ChildItem -LiteralPath $dst -Filter 'api.*.log' -ErrorAction SilentlyContinue).Count) + ' 个API日志文件') }"

echo.
echo 采集会话数据...
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $src='%SESSIONS_DIR%'; $dst='%STAGE%\sessions'; $start=[datetime]::Parse('%START_TIME%'); $end=[datetime]::Parse('%END_TIME%'); $cnt=0; if (Test-Path $src) { Get-ChildItem -LiteralPath $src -Directory -Filter 'officeclaw_*' | Where-Object { $_.LastWriteTime -ge $start -and $_.LastWriteTime -le $end } | ForEach-Object { $out = Join-Path $dst $_.Name; if (-not (Test-Path $out)) { New-Item -ItemType Directory -Path $out -Force | Out-Null }; $h = Join-Path $_.FullName 'history.json'; $m = Join-Path $_.FullName 'metadata.json'; if (Test-Path $h) { Copy-Item $h $out -Force; $cnt++ }; if (Test-Path $m) { Copy-Item $m $out -Force } } }; Write-Host ('已采集 ' + $cnt + ' 个会话') }"

echo.
echo 生成清单文件...
(
echo OfficeClaw 日志采集清单
echo 版本：4.2.0
echo 时间范围：%START_TIME% ~ %END_TIME%
echo 安装目录：%INSTALL_DIR%
echo 数据目录：%BASE_DIR%
echo 日志目录：%LOGS_DIR%
echo 会话目录：%SESSIONS_DIR%
) > "%STAGE%\MANIFEST.txt"

echo.
echo 打包中...
set "ZIP=%OUTPUT_DIR%\%BUNDLE_NAME%.zip"
if exist "%ZIP%" del "%ZIP%"
powershell -NoProfile -Command "Compress-Archive -Path '%STAGE%' -DestinationPath '%ZIP%' -Force"

if exist "%ZIP%" (
    echo.
    echo ========================================
    echo 采集完成！
    echo ========================================
    echo 输出文件：%ZIP%
) else (
    echo 错误：打包失败
)

rd /s /q "%TEMP_DIR%" 2>nul
echo.
pause
exit /b 0
