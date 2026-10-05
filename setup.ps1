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
    [string[]]$AppArgs = @()
)

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$taskRoot = $PSScriptRoot

function Test-FastVideoPython([string]$Executable) {
    try {
        & $Executable -c 'import sys, tkinter, venv; sys.exit(sys.version_info[:2] != (3, 12))' 2>$null | Out-Null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

try {
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
                & $taskWindowsPowerShell -NoProfile -ExecutionPolicy Bypass -File $taskInstaller
                if ($LASTEXITCODE -ne 0) { throw 'The uv installation failed.' }
            }
        }
        & $taskUv python install 3.12
        if ($LASTEXITCODE -ne 0) { throw 'The Python 3.12 installation failed.' }
        $Python = & $taskUv python find --python-preference only-managed 3.12
        if ($LASTEXITCODE -ne 0 -or -not (Test-FastVideoPython $Python)) {
            throw 'Python 3.12 could not load Tkinter. See docs/installation.md.'
        }
    }

    $Python = & $Python -c 'import sys; print(sys.executable)'
    if ($LASTEXITCODE -ne 0) { throw 'The Python interpreter could not start.' }
    $taskArguments = @('-m', 'tools.bootstrap', '--backend', $Backend, '--amd-arch', $AmdArch, '--model', $Model)
    if ($Venv) { $taskArguments += @('--venv', $Venv) }
    if ($SetupOnly) { $taskArguments += '--setup-only' }
    if ($DryRun) { $taskArguments += '--dry-run' }
    if ($AppArgs.Count -gt 0) { $taskArguments += @('--') + $AppArgs }
    Push-Location -LiteralPath $taskRoot
    try {
        & $Python @taskArguments
        $taskExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    exit $taskExitCode
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
}
