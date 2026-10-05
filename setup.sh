#!/usr/bin/env bash
# First clone: Python 3.12, local environment, GPU dependencies, then FastVideo.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROOT"
export PYTHONUTF8=1
export PYTHONDONTWRITEBYTECODE=1

is_python() {
    "$1" -c 'import sys, tkinter, venv; sys.exit(sys.version_info[:2] != (3, 12))' \
        >/dev/null 2>&1
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

exec "$PYTHON" -m tools.bootstrap "$@"
