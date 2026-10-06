#!/usr/bin/env bash
# First clone: Python 3.12, local environment, GPU dependencies, then FastVideo.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"

main() {
    export PYTHONUTF8=1
    export PYTHONDONTWRITEBYTECODE=1

    is_python() {
        local probe_output probe_code
        if probe_output=$("$1" -c 'import sys, tkinter, venv; sys.exit(sys.version_info[:2] != (3, 12))' 2>&1); then
            return 0
        else
            probe_code=$?
            printf 'Python probe failed: %s (code %s)\n%s\n' "$1" "$probe_code" "$probe_output"
            return "$probe_code"
        fi
    }

    PYTHON="${FASTVIDEO_PYTHON:-}"
    if [[ -n "$PYTHON" ]]; then
        if ! is_python "$PYTHON"; then
            echo "FASTVIDEO_PYTHON must point to Python 3.12 with Tkinter and venv." >&2
            exit 1
        fi
    else
        for candidate in "$ROOT/.venv/bin/python" "$ROOT/.venv/Scripts/python.exe" \
            python3.12 python3 python; do
            if is_python "$candidate"; then
                PYTHON="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$PYTHON" ]]; then
        for arg in "$@"; do
            if [[ "$arg" == "--dry-run" || "$arg" == "--help" || "$arg" == "-h" ]]; then
                echo "Python 3.12 with Tkinter is missing; a normal run installs it in .tools/."
                echo "Usage: bash setup.sh [--backend auto|cuda|rocm|cpu] [--amd-arch gfx1150]"
                echo "                     [--model smol] [--setup-only] [--dry-run] [-- APP_OPTIONS]"
                exit 0
            fi
        done
        export UV_INSTALL_DIR="$ROOT/.tools/uv"
        export UV_NO_MODIFY_PATH=1
        export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
        export UV_PYTHON_BIN_DIR="$ROOT/.tools/python-bin"
        export UV_CACHE_DIR="$ROOT/.tools/uv-cache"
        export UV_PYTHON_INSTALL_BIN=0
        export UV_PYTHON_INSTALL_REGISTRY=0
        UV="$UV_INSTALL_DIR/uv"
        if [[ ! -x "$UV" ]]; then
            if command -v uv >/dev/null 2>&1; then
                UV="$(command -v uv)"
            else
                mkdir -p -- "$UV_INSTALL_DIR"
                INSTALLER="$UV_INSTALL_DIR/install.sh"
                URL="https://astral.sh/uv/0.12.23/install.sh"
                if command -v curl >/dev/null 2>&1; then
                    curl --fail --location --silent --show-error "$URL" -o "$INSTALLER"
                elif command -v wget >/dev/null 2>&1; then
                    wget -q "$URL" -O "$INSTALLER"
                else
                    echo "Install curl or wget, then run this script again." >&2
                    exit 1
                fi
                sh "$INSTALLER"
            fi
        fi
        "$UV" python install 3.12
        PYTHON="$("$UV" python find --python-preference only-managed 3.12)"
        if ! is_python "$PYTHON"; then
            echo "Python 3.12 could not load Tkinter. See docs/installation.md." >&2
            exit 1
        fi
    fi

    printf 'Python: %s\n' "$PYTHON"
    "$PYTHON" -u -m tools.bootstrap "$@"
}

task_log_name="startup-$(date +%Y%m%d-%H%M%S)-$$.log"
task_log_dir="$ROOT/logs"
if ! mkdir -p -- "$task_log_dir" || ! : > "$task_log_dir/startup-latest.log"; then
    task_log_dir="${TMPDIR:-/tmp}/fastvideo-logs"
    mkdir -p -- "$task_log_dir"
    : > "$task_log_dir/startup-latest.log"
fi
task_log="$task_log_dir/$task_log_name"
task_latest_log="$task_log_dir/startup-latest.log"
printf 'Startup log: %s\n' "$task_latest_log"
set +e
(
    set -eE
    trap 'task_exit=$?; printf "End: %s - Exit code: %s\nStartup log: %s\n" "$(date -Iseconds)" "$task_exit" "$task_latest_log"' EXIT
    trap 'printf "Launcher failed at line %s (code %s)\n" "$LINENO" "$?" >&2' ERR
    printf 'Start: %s\nBash: %s\nProject: %s\n' "$(date -Iseconds)" "$BASH_VERSION" "$ROOT"
    main "$@"
) 2>&1 | tee -a "$task_log" "$task_latest_log"
task_codes=("${PIPESTATUS[@]}")
if [[ "${task_codes[0]}" -ne 0 ]]; then
    exit "${task_codes[0]}"
fi
exit "${task_codes[1]}"
