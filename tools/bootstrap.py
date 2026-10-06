"""First-clone installation shared by setup.sh and setup.ps1 (standard library only)."""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from vision import DEFAULT_MODEL, MODELS

ROOT = Path(__file__).resolve().parents[1]
AMD_ARCHES = ("gfx1150", "gfx1151", "gfx1152", "gfx1100", "gfx1201")
WHEELS = {
    "cuda": ("https://download.pytorch.org/whl/cu128", "2.11.0+cu128", "0.26.0+cu128"),
    "cpu": ("https://download.pytorch.org/whl/cpu", "2.11.0+cpu", "0.26.0+cpu"),
    "rocm": (
        "https://stable.repo.amd.com/rocm/whl-next/",
        "2.13.0+rocm10.0.0",
        "0.28.0+rocm10.0.0",
    ),
}
PROBE = """
import importlib, json, sys
from importlib.metadata import version
report = {'python': list(sys.version_info[:2]), 'errors': [], 'backend': None,
          'available': False, 'arch': None}
for module, package in [('tkinter', None), ('torch', 'torch'), ('torchvision', 'torchvision'),
                        ('cv2', 'opencv-python'), ('PIL', 'Pillow'),
                        ('transformers', 'transformers'), ('timm', 'timm'),
                        ('num2words', 'num2words')]:
    try:
        imported = importlib.import_module(module)
        if package:
            report[package] = version(package)
        if module == 'torch':
            report['backend'] = 'rocm' if imported.version.hip else (
                'cuda' if imported.version.cuda else 'cpu')
            report['available'] = imported.cuda.is_available()
            if report['available']:
                report['gpu'] = imported.cuda.get_device_name(0)
                report['arch'] = getattr(imported.cuda.get_device_properties(0),
                                         'gcnArchName', '').split(':')[0] or None
    except Exception as error:
        report['errors'].append(f'{module}: {error}')
print(json.dumps(report))
"""


def python_in(venv):
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def probe(python):
    if not python.is_file():
        return {}
    result = subprocess.run(
        [str(python), "-c", PROBE], capture_output=True, text=True, encoding="utf-8", timeout=90
    )
    if result.stderr.strip():
        print(f"Diagnostic du runtime {python}:\n{result.stderr}", flush=True)
    if result.returncode:
        raise RuntimeError(
            f"L'environnement {python.parent.parent} ne démarre pas.\n{result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def amd_hardware(names):
    arch = next((arch for arch in AMD_ARCHES if arch in names), None)
    # Priorité au target HIP / rocminfo, puis au GPU, enfin au nom du CPU.
    # 840M existe sur deux architectures : ne pas déduire son target seul.
    if arch is None:
        for pattern, target in (
            (r"\b(?:8065S|8060S|8050S|8040S)\b", "gfx1151"),
            (r"\b(?:890M|880M)\b", "gfx1150"),
            (r"\b(?:860M|820M)\b", "gfx1152"),
            (
                r"\bRyzen\s+AI\s+Max\+?\s+(?:PRO\s+)?(?:380|385|388|390|392|395|485|490|495)\b",
                "gfx1151",
            ),
            (r"\bHX\s+(?:PRO\s+)?(?:370|375|470|475)\b|\bHX(?:370|375|470|475)\b", "gfx1150"),
            (r"\bRyzen\s+AI\s+9\s+(?:PRO\s+)?(?:365|465)\b", "gfx1150"),
            (r"\bRyzen\s+AI\s+[57]\s+(?:PRO\s+)?(?:330|340|345|350|440|450)\b", "gfx1152"),
        ):
            if re.search(pattern, names, re.IGNORECASE):
                arch = target
                break
    amd = bool(
        arch
        or re.search(
            r"Radeon|AMD.*(?:Graphics|VGA|Display)|(?:VGA|Display|3D).*AMD", names, re.IGNORECASE
        )
    )
    return amd, arch


def hardware():
    nvidia = "GPU " in command_output(["nvidia-smi", "-L"])
    if os.name == "nt":
        names = command_output(
            [
                str(
                    Path(os.environ.get("SystemRoot", r"C:\Windows"))
                    / "System32/WindowsPowerShell/v1.0/powershell.exe"
                ),
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name; "
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
    else:
        cpu = Path("/proc/cpuinfo")
        names = (cpu.read_text(errors="replace") if cpu.exists() else "") + command_output(
            ["lspci"]
        )
        names += command_output(["rocminfo"])
    amd, arch = amd_hardware(names)
    return {"nvidia": nvidia, "amd": amd, "arch": arch}


def select_backend(requested, requested_arch, installed, detected):
    backend = requested
    if backend == "auto":
        if requested_arch != "auto":
            backend = "rocm"
        elif installed.get("available") and installed.get("backend") in ("cuda", "rocm"):
            backend = installed["backend"]
        elif detected["nvidia"]:
            backend = "cuda"
        else:
            backend = "rocm" if detected["amd"] else "cpu"
    arch = None
    if backend == "rocm":
        arch = (
            requested_arch
            if requested_arch != "auto"
            else (installed.get("arch") or detected["arch"])
        )
        if arch not in AMD_ARCHES:
            raise RuntimeError(
                "Architecture AMD inconnue. Indique --amd-arch (PowerShell : -AmdArch) : "
                "gfx1150 pour HX370/HX470, gfx1151 pour Ryzen AI Max 385/395. "
                "Voir docs/installation.md pour les prérequis AMD."
            )
    return backend, arch


def compatible(installed, backend, arch):
    def at_least(name, minimum):
        numbers = re.match(r"(\d+)\.(\d+)", installed.get(name, ""))
        return bool(numbers and tuple(map(int, numbers.groups())) >= minimum)

    return (
        installed.get("backend") == backend
        and at_least("torch", (2, 8))
        and at_least("torchvision", (0, 23))
        and (backend == "cpu" or installed.get("available"))
        and (backend != "rocm" or installed.get("arch") == arch)
        and not any(
            error.startswith(("torch:", "torchvision:")) for error in installed.get("errors", [])
        )
    )


def torch_command(python, backend, arch):
    index, torch, torchvision = WHEELS[backend]
    extra = f"[device-{arch}]" if backend == "rocm" else ""
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--index-url",
        index,
        f"torch{extra}=={torch}",
        f"torchvision{extra}=={torchvision}",
    ]


def run(command):
    print(f"\n> {shlex.join(map(str, command))}", flush=True)
    started = time.monotonic()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        print(
            f"Commande échouée · code {error.returncode} · {time.monotonic() - started:.1f} s",
            flush=True,
        )
        raise
    else:
        print(f"Commande terminée · code 0 · {time.monotonic() - started:.1f} s", flush=True)


def setup(venv, backend, arch, installed, dry_run):
    python = python_in(venv)
    requirements = ROOT / "requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes() + Path(__file__).read_bytes()).hexdigest()
    marker = venv / ".fastvideo-setup.json"
    expected = {"signature": signature, "backend": backend, "arch": arch}
    try:
        saved = json.loads(marker.read_text())
    except (OSError, ValueError):
        saved = {}
    ready = (
        saved == expected and compatible(installed, backend, arch) and not installed.get("errors")
    )
    constraints = venv / ".fastvideo-torch.txt"
    dependencies = [
        str(python),
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements),
        "-c",
        str(constraints),
    ]
    if dry_run:
        if not python.is_file():
            print("> " + shlex.join([sys.executable, "-m", "venv", str(venv)]))
        if not compatible(installed, backend, arch):
            print("> " + shlex.join(torch_command(python, backend, arch)))
        if not ready:
            print("Contraintes : conserver les versions PyTorch / torchvision choisies.")
            print("> " + shlex.join(dependencies))
        print("Vérification des dépendances, de Tkinter et du GPU avant le lancement.")
        return python
    if not python.is_file():
        if venv.exists():
            raise RuntimeError(
                f"{venv} existe mais ne contient pas de Python. Choisis un autre --venv."
            )
        run([sys.executable, "-m", "venv", str(venv)])
    if not ready:
        if not compatible(installed, backend, arch):
            run(torch_command(python, backend, arch))
            installed = probe(python)
        if not compatible(installed, backend, arch):
            raise RuntimeError(
                f"PyTorch {backend.upper()} ne voit pas le GPU attendu. "
                "Vérifie les pilotes et les prérequis dans docs/installation.md, puis relance le script."
            )
        constraints.write_text(
            f"torch=={installed['torch']}\ntorchvision=={installed['torchvision']}\n",
            encoding="utf-8",
        )
        run(dependencies)
        installed = probe(python)
        if installed["errors"] or not compatible(installed, backend, arch):
            raise RuntimeError("Vérification impossible :\n" + "\n".join(installed["errors"]))
        marker.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Environnement prêt : {backend.upper()} · {installed.get('gpu', 'processeur')}")
    return python


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Installer et lancer FastVideo après le premier clone."
    )
    parser.add_argument("--backend", choices=("auto", "cuda", "rocm", "cpu"), default="auto")
    parser.add_argument("--amd-arch", choices=("auto", *AMD_ARCHES), default="auto")
    parser.add_argument("--model", choices=[model.key for model in MODELS], default=DEFAULT_MODEL)
    parser.add_argument("--setup-only", action="store_true", help="Installer sans ouvrir l'app.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Afficher le plan sans rien installer."
    )
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv")
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="Options de l'app après --.")
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12):
        parser.error("Python 3.12 requis ; utilise setup.sh ou setup.ps1.")
    if platform.system() not in ("Windows", "Linux"):
        parser.error("L'installation automatique prend en charge Windows et Linux.")
    venv = args.venv.resolve()
    print(f"Système : {platform.platform()} · {platform.machine()}", flush=True)
    print(f"Python : {sys.version.split()[0]} · {sys.executable}", flush=True)
    print(f"Projet : {ROOT}\nEnvironnement : {venv}", flush=True)
    installed = probe(python_in(venv))
    print("Runtime existant : " + json.dumps(installed, ensure_ascii=False), flush=True)
    if installed and installed["python"] != [3, 12]:
        parser.error(
            f"{venv} utilise un autre Python. "
            "Choisis un nouveau dossier avec --venv (PowerShell : -Venv)."
        )
    detected = hardware()
    print("Matériel détecté : " + json.dumps(detected, ensure_ascii=False), flush=True)
    backend, arch = select_backend(args.backend, args.amd_arch, installed, detected)
    print(
        f"FastVideo · Python 3.12 · {backend.upper()}" + (f" · {arch}" if arch else ""), flush=True
    )
    python = setup(venv, backend, arch, installed, args.dry_run)
    if not args.setup_only:
        extra = args.app_args[1:] if args.app_args[:1] == ["--"] else args.app_args
        command = [
            str(python),
            "-u",
            str(ROOT / "app.py"),
            "--model",
            args.model,
            "--device",
            backend,
            *extra,
        ]
        if args.dry_run:
            print("> " + shlex.join(command))
        else:
            run(command)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"\nCommande interrompue (code {error.returncode}). Relance après correction.",
            file=sys.stderr,
        )
        sys.exit(error.returncode or 1)
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"\nInstallation interrompue : {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
