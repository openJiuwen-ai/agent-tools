@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   OfficeClaw 日志采集工具
echo ========================================
echo.

echo 请输入采集日志的时间范围
echo.

:input_start
set "START_TIME="
set /p "START_TIME=请输入开始时间（例如：2026-05-07 00:00:00）: "
if "!START_TIME!"=="" (
    echo 错误：开始时间不能为空
    goto :input_start
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { try { [datetime]::Parse('%START_TIME%') } catch { Write-Host '错误：时间格式不正确，请使用格式：YYYY-MM-DD HH:MM:SS'; exit 1 } }"
if errorlevel 1 goto :input_start

:input_end
set "END_TIME="
set /p "END_TIME=请输入结束时间（例如：2026-05-07 23:59:59）: "
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

rem 在括号块外解析，避免 (x86) 中的 ) 破坏 if 块解析
set "PROGRAM_FILES_X86=%ProgramFiles(x86)%"

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
    if exist "!PROGRAM_FILES_X86!\OfficeClaw\OfficeClaw.exe" (
        set "INSTALL_DIR=!PROGRAM_FILES_X86!\OfficeClaw"
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
    if exist "!INSTALL_DIR!\OfficeClaw.exe" (
        echo 找到安装目录：!INSTALL_DIR!
        echo.
    ) else (
        set "INSTALL_DIR="
    )
)

if not defined INSTALL_DIR (
    echo 未能自动找到OfficeClaw安装目录，请手动输入：
    set /p "INSTALL_DIR=安装目录路径: "
    if not exist "!INSTALL_DIR!\OfficeClaw.exe" (
        echo 错误：指定的目录不是有效的OfficeClaw安装目录
        pause
        exit /b 1
    )
)

set "OUTPUT_DIR=%~dp0"

echo.
echo 正在采集...
echo.

rem chcp 65001 会改变 %%date%% 格式，改用 PowerShell 生成稳定时间戳
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "BUNDLE_NAME=office_claw_log_%%i"

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
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $dst='%STAGE%\runtime_logs'; $start=[datetime]::Parse('%START_TIME%'); $end=[datetime]::Parse('%END_TIME%'); $total = (Get-ChildItem -LiteralPath $dst -Recurse -File).Count; Write-Host ('过滤前文件数: ' + $total); $removed=0; Get-ChildItem -LiteralPath $dst -Recurse -File | ForEach-Object { $keep = $false; if ($_.Name -match '_(\d{8}_\d{6})\.log$') { $fileDate = [datetime]::ParseExact($Matches[1], 'yyyyMMdd_HHmmss', $null); if ($fileDate -ge $start -and $fileDate -le $end) { $keep = $true } } else { if ($_.LastWriteTime -ge $start) { $keep = $true } }; if (-not $keep) { Remove-Item -LiteralPath $_.FullName -Force; $removed++ } }; Write-Host ('已移除 ' + $removed + ' 个不在时间范围内的文件'); $remaining = (Get-ChildItem -LiteralPath $dst -Recurse -File).Count; Write-Host ('过滤后文件数: ' + $remaining) }"

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
echo 版本：4.0.0
echo 时间范围：%START_TIME% ~ %END_TIME%
echo 安装目录：!INSTALL_DIR!
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
