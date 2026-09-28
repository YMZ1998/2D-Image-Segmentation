@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PIXI_ENV=%PROJECT_DIR%.pixi\envs\default"

if not exist "%PIXI_ENV%\python.exe" (
    echo.
    echo [ERROR] Pixi environment not found:
    echo %PIXI_ENV%
    echo.
    pause
    exit /b 1
)

set "PATH=%PIXI_ENV%;%PIXI_ENV%\Scripts;%PATH%"
set "PROMPT=(pixi) $P$G"

echo.
echo ========================================
echo  Pixi environment activated
echo ========================================
echo  Environment: %PIXI_ENV%
echo.

cmd /k