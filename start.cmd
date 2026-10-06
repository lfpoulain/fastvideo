@echo off
setlocal
rem Double-click launcher: setup.ps1 writes logs/startup-latest.log automatically.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -NoPause %*
set "taskExitCode=%ERRORLEVEL%"
if not "%taskExitCode%"=="0" (
    echo.
    echo Le lancement a echoue. Consulte logs\startup-latest.log dans le dossier FastVideo.
    pause
)
exit /b %taskExitCode%
