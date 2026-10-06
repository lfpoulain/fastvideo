# First clone: Python 3.12, local environment, GPU dependencies, then FastVideo.
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cuda', 'rocm', 'cpu')]
    [string]$Backend = 'auto',
    [string]$AmdArch = 'auto',
    [string]$Model = 'smol',
    [switch]$SetupOnly,
    [switch]$DryRun,
    [string]$Python = $env:FASTVIDEO_PYTHON,
    [string]$Venv,
    [string]$LogFile,
    [switch]$NoPause,
    [string[]]$AppArgs = @()
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskRoot = $PSScriptRoot
$taskExitCode = 1
$taskTranscriptStarted = $false
$taskArchiveLog = $null

function Invoke-FastVideoNative([string]$Executable, [string[]]$Arguments) {
    # Windows PowerShell 5 treats redirected native stderr as ErrorRecords.
    # Keep stderr in the transcript without aborting a successful native command.
    $taskPreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments 2>&1 | Out-Host
        $script:taskNativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $taskPreviousPreference
    }
}

function Test-FastVideoPython([string]$Executable) {
    $taskPreviousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $taskProbeOutput = & $Executable -c 'import sys, tkinter, venv; sys.exit(sys.version_info[:2] != (3, 12))' 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Python probe failed: $Executable"
            $taskProbeOutput | Out-Host
        }
        return $LASTEXITCODE -eq 0
    } catch {
        Write-Host "Python unavailable: $Executable - $($_.Exception.Message)"
        return $false
    } finally {
        $ErrorActionPreference = $taskPreviousPreference
    }
}

try {
    $taskDefaultLog = -not $LogFile
    if ($taskDefaultLog) {
        $taskLogName = 'startup-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + "-$PID.log"
        $LogFile = Join-Path $taskRoot 'logs\startup-latest.log'
    }
    $LogFile = [IO.Path]::GetFullPath($LogFile)
    try {
        New-Item -ItemType Directory -Path (Split-Path -Parent $LogFile) -Force | Out-Null
        Start-Transcript -LiteralPath $LogFile -Force | Out-Null
    } catch {
        if (-not $taskDefaultLog) { throw }
        $LogFile = Join-Path ([IO.Path]::GetTempPath()) 'FastVideo\logs\startup-latest.log'
        New-Item -ItemType Directory -Path (Split-Path -Parent $LogFile) -Force | Out-Null
        Start-Transcript -LiteralPath $LogFile -Force | Out-Null
    }
    $taskTranscriptStarted = $true
    if ($taskDefaultLog) {
        $taskArchiveLog = Join-Path (Split-Path -Parent $LogFile) $taskLogName
    }
    Write-Host "Startup log: $LogFile"
    Write-Host "Start: $(Get-Date -Format o)"
    Write-Host "PowerShell: $($PSVersionTable.PSVersion) - OS: $([Environment]::OSVersion)"
    Write-Host "Project: $taskRoot - Backend: $Backend - AMD: $AmdArch - Model: $Model"
    if ($Python) {
        if (-not (Test-FastVideoPython $Python)) {
            throw '-Python / FASTVIDEO_PYTHON must point to Python 3.12 with Tkinter and venv.'
        }
    } else {
        $taskCandidates = @((Join-Path $taskRoot '.venv\Scripts\python.exe'), 'python3.12', 'python')
        foreach ($taskCandidate in $taskCandidates) {
            if (Test-FastVideoPython $taskCandidate) {
                $Python = $taskCandidate
                break
            }
        }
        if (-not $Python -and (Get-Command py -ErrorAction SilentlyContinue)) {
            try {
                $taskCandidate = & py -3.12 -c 'import sys; print(sys.executable)' 2>$null
                if ($LASTEXITCODE -eq 0 -and (Test-FastVideoPython $taskCandidate)) {
                    $Python = $taskCandidate
                }
            } catch { $Python = '' }
        }
    }

    if (-not $Python) {
        if ($DryRun) {
            Write-Host 'Python 3.12 with Tkinter is missing; a normal run installs it in .tools/.'
            $taskExitCode = 0
            exit 0
        }
        $env:UV_INSTALL_DIR = Join-Path $taskRoot '.tools\uv'
        $env:UV_NO_MODIFY_PATH = '1'
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRoot '.tools\python'
        $env:UV_PYTHON_BIN_DIR = Join-Path $taskRoot '.tools\python-bin'
        $env:UV_CACHE_DIR = Join-Path $taskRoot '.tools\uv-cache'
        $env:UV_PYTHON_INSTALL_BIN = '0'
        $env:UV_PYTHON_INSTALL_REGISTRY = '0'
        $taskUv = Join-Path $env:UV_INSTALL_DIR 'uv.exe'
        if (-not (Test-Path -LiteralPath $taskUv)) {
            $taskGlobalUv = Get-Command uv -ErrorAction SilentlyContinue
            if ($taskGlobalUv) {
                $taskUv = $taskGlobalUv.Source
            } else {
                New-Item -ItemType Directory -Path $env:UV_INSTALL_DIR -Force | Out-Null
                $taskInstaller = Join-Path $env:UV_INSTALL_DIR 'install.ps1'
                Invoke-WebRequest -UseBasicParsing 'https://astral.sh/uv/0.12.23/install.ps1' -OutFile $taskInstaller
                $taskWindowsPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
                Invoke-FastVideoNative -Executable $taskWindowsPowerShell -Arguments @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $taskInstaller)
                if ($taskNativeExitCode -ne 0) { throw 'The uv installation failed.' }
            }
        }
        Invoke-FastVideoNative -Executable $taskUv -Arguments @('python', 'install', '3.12')
        if ($taskNativeExitCode -ne 0) { throw 'The Python 3.12 installation failed.' }
        $Python = & $taskUv python find --python-preference only-managed 3.12
        if ($LASTEXITCODE -ne 0 -or -not (Test-FastVideoPython $Python)) {
            throw 'Python 3.12 could not load Tkinter. See docs/installation.md.'
        }
    }

    $Python = & $Python -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'The Python interpreter could not start.' }
    Write-Host "Python: $Python"
    $taskArguments = @('-u', '-m', 'tools.bootstrap', '--backend', $Backend, '--amd-arch', $AmdArch, '--model', $Model)
    if ($Venv) { $taskArguments += @('--venv', $Venv) }
    if ($SetupOnly) { $taskArguments += '--setup-only' }
    if ($DryRun) { $taskArguments += '--dry-run' }
    if ($AppArgs.Count -gt 0) { $taskArguments += @('--') + $AppArgs }
    Push-Location -LiteralPath $taskRoot
    try {
        Invoke-FastVideoNative -Executable $Python -Arguments $taskArguments
        $taskExitCode = $taskNativeExitCode
    } finally {
        Pop-Location
    }
    exit $taskExitCode
} catch {
    $taskExitCode = 1
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    Write-Host "End: $(Get-Date -Format o) - Exit code: $taskExitCode"
    Write-Host "Startup log: $LogFile"
    if ($taskTranscriptStarted) { Stop-Transcript | Out-Null }
    if ($taskArchiveLog) {
        try {
            Copy-Item -LiteralPath $LogFile -Destination $taskArchiveLog -Force
        } catch {
            Write-Host "Could not archive startup log: $($_.Exception.Message)"
        }
    }
    if ($taskExitCode -ne 0 -and -not $NoPause -and $Host.Name -eq 'ConsoleHost') {
        try {
            if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
                Read-Host 'Echec du lancement. Appuyez sur Entree pour fermer' | Out-Null
            }
        } catch { }
    }
}
