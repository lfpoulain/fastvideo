"""Vision sur NPU AMD avec un processus FastFlowLM privé, sans dépendance Python."""

import atexit
import base64
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import replace
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from vision import DEFAULT_PROMPT, MODEL_BY_KEY

NPU_MODELS = {
    "qwen-0.8b": "qwen3.5:0.8b",
    "qwen-2b": "qwen3.5:2b",
    "qwen-4b": "qwen3.5:4b",
}
NPU_INSTALL_URL = "https://fastflowlm.com/docs/install_win/"


def windows_flm_locations():
    """Lire le registre actuel, même si le processus a hérité d'un ancien PATH."""
    try:
        import winreg
    except ImportError:
        return []

    locations = []

    def value(key, name):
        try:
            return winreg.QueryValueEx(key, name)[0]
        except OSError:
            return ""

    for hive, environment in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ):
        try:
            with winreg.OpenKey(hive, environment) as key:
                locations.extend(str(value(key, "Path")).split(";"))
        except OSError:
            pass
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            access = winreg.KEY_READ | view
            try:
                with winreg.OpenKey(
                    hive,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\flm.exe",
                    0,
                    access,
                ) as key:
                    locations.append(value(key, ""))
            except OSError:
                pass
            try:
                with winreg.OpenKey(
                    hive,
                    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
                    0,
                    access,
                ) as uninstall:
                    index = 0
                    while True:
                        try:
                            name = winreg.EnumKey(uninstall, index)
                        except OSError:
                            break
                        index += 1
                        try:
                            with winreg.OpenKey(uninstall, name) as key:
                                if not re.search(
                                    r"fastflowlm|\bflm\b", str(value(key, "DisplayName")), re.I
                                ):
                                    continue
                                locations.extend(
                                    (
                                        value(key, "InstallLocation"),
                                        value(key, "Inno Setup: App Path"),
                                    )
                                )
                                icon = str(value(key, "DisplayIcon"))
                                match = re.match(
                                    r'^"([^"]+\.exe)"|^(.+?\.exe)(?:,\s*-?\d+)?$', icon, re.I
                                )
                                if match:
                                    locations.append(match.group(1) or match.group(2))
                        except OSError:
                            continue
            except OSError:
                pass
    return [location for location in locations if isinstance(location, str) and location.strip()]


def flm_candidates(location):
    root = Path(os.path.expandvars(str(location).strip().strip('"'))).expanduser()
    if root.name.lower() in ("flm.exe", "flm") and root.is_file():
        yield root
    for name in ("flm.exe", "bin/flm.exe", "flm", "bin/flm"):
        yield root / name


def find_flm(filename=None):
    filename = filename or os.environ.get("FASTVIDEO_FLM_PATH")
    if filename:
        candidate = (
            Path(os.path.expandvars(str(filename).strip().strip('"'))).expanduser().resolve()
        )
        if candidate.is_file():
            return str(candidate)
        for executable in flm_candidates(candidate):
            if executable.is_file():
                return str(executable.resolve())
        raise RuntimeError(f"Exécutable FastFlowLM introuvable : {candidate}")
    if executable := shutil.which("flm"):
        return executable

    locations = windows_flm_locations() if sys.platform == "win32" else []
    for variable in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        root = os.environ.get(variable)
        if root:
            locations.extend(
                Path(root) / name
                for name in ("flm", "FastFlowLM", "Programs/flm", "Programs/FastFlowLM")
            )
    if sys.platform != "win32":
        locations.extend(("/opt/fastflowlm", "~/.local/bin"))
    seen = set()
    for location in locations:
        for candidate in flm_candidates(location):
            identity = os.path.normcase(str(candidate))
            if identity in seen:
                continue
            seen.add(identity)
            if candidate.is_file():
                return str(candidate.resolve())
    raise RuntimeError(
        "FastFlowLM introuvable dans le PATH et les emplacements d’installation. "
        "S’il est déjà installé, clique sur « Choisir flm.exe… ». "
        "Sinon, utilise « Installer FastFlowLM… »."
    )


class NpuVision:
    def __init__(self, key, offline=False, progress=None, executable=None, cancel=None):
        if key not in NPU_MODELS:
            raise ValueError(
                "Ce modèle n’a pas de version NPU dans FastVideo. "
                "Choisis Qwen3.5 0,8B, 2B ou 4B, ou utilise le GPU/CPU."
            )
        # FLM ne fournit pas de flag garantissant l'absence de métadonnées réseau.
        if offline:
            raise ValueError(
                "Le mode --offline strict n’est pas garanti par FastFlowLM. "
                "Utilise le GPU/CPU pour ce mode. L’analyse NPU reste locale."
            )
        self.spec = replace(MODEL_BY_KEY[key], max_frames=1)
        self.tag = NPU_MODELS[key]
        self.device = "npu"
        self.backend = "NPU AMD · FastFlowLM"
        self.progress, self.cancel = progress, cancel
        self.process = None
        self.command_process = None
        self.reader = None
        self._close_lock = threading.Lock()
        self.closed = False
        self.output = deque(maxlen=30)
        self.diagnostics = [f"Runtime · FastFlowLM · {self.tag} · Q4 · dernière image uniquement"]
        # Pas de proxy système ni de redirection HTTP pour les images de webcam.
        self.opener = build_opener(ProxyHandler({}), LocalRedirectHandler())
        started = time.perf_counter()
        self._emit("check", "Recherche de l’installation FastFlowLM…")
        self.executable = find_flm(executable)
        # La fenêtre peut être fermée pendant une préparation dans un thread
        # daemon. Arrêter aussi les processus enfants à la sortie de Python.
        atexit.register(self.close)
        try:
            self._emit("diagnostic", f"FastFlowLM détecté · {self.executable}")
            self._emit("check", "Vérification du NPU AMD et de son pilote…")
            self._command(["validate"], timeout=60)
            self._emit("load", f"Préparation NPU · {self.tag} · téléchargement si nécessaire…")
            self._command(["pull", self.tag], timeout=3600)
            self._emit("load", f"Chargement NPU · {self.spec.label}")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            self._start_server(
                [
                    "serve",
                    self.tag,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--ctx-len",
                    "8192",
                    "-r",
                    "0",
                    "--cors",
                    "0",
                    "--quiet",
                ]
            )
            self._wait_ready(timeout=300)
            for message in self.diagnostics:
                self._emit("diagnostic", message)
            self._emit(
                "ready",
                f"Modèle prêt · {self.backend} · {time.perf_counter() - started:.1f} s",
                backend=self.backend,
                dtype="Q4",
            )
        except BaseException:
            self.close()
            raise

    def _emit(self, stage, message, **data):
        if self.progress:
            self.progress(dict(stage=stage, message=message, **data))

    def _spawn(self, arguments):
        return subprocess.Popen(
            [self.executable, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def _start_server(self, arguments):
        with self._close_lock:
            if self.closed:
                raise RuntimeError("Moteur NPU fermé.")
            self.process = self._spawn(arguments)
            # Seule la préparation est remontée ; aucun log d'inférence n'est conservé.
            self.reader = threading.Thread(
                target=self._drain, args=(self.process, False), daemon=True
            )
            self.reader.start()

    def _drain(self, process, forward):
        last = 0.0
        for line in process.stdout:
            line = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line).strip()[:1000]
            if not line:
                continue
            if forward:
                self.output.append(line)
                now = time.monotonic()
                if now - last >= 0.5:
                    self._emit("diagnostic", "FastFlowLM · " + line)
                    last = now
            # La sortie du serveur est volontairement ignorée : certains moteurs
            # affichent les prompts même en mode quiet.

    @staticmethod
    def _stop(process):
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def _command(self, arguments, timeout):
        self.output.clear()
        with self._close_lock:
            if self.closed:
                raise RuntimeError("Moteur NPU fermé.")
            process = self.command_process = self._spawn(arguments)
        reader = threading.Thread(target=self._drain, args=(process, True), daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if self.cancel is not None and self.cancel.is_set():
                    raise RuntimeError("Préparation NPU annulée.")
                if time.monotonic() >= deadline:
                    raise RuntimeError("Délai de préparation NPU dépassé.")
                time.sleep(0.1)
            reader.join(timeout=2)
            if process.returncode:
                detail = "\n".join(self.output)
                raise RuntimeError(
                    f"FastFlowLM {arguments[0]} a échoué (code {process.returncode}). "
                    f"Vérifie le pilote NPU et la version de FastFlowLM.\n{detail}"
                )
        finally:
            self._stop(process)
            reader.join(timeout=2)
            process.stdout.close()
            self.command_process = None

    def _request(self, path, payload=None, timeout=600):
        request = Request(
            self.url + path,
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
            headers={"Content-Type": "application/json"},
        )
        return self.opener.open(request, timeout=timeout)

    def _wait_ready(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.cancel is not None and self.cancel.is_set():
                raise RuntimeError("Chargement NPU annulé.")
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Le moteur NPU s’est arrêté (code {self.process.returncode}). "
                    "Vérifie le pilote AMD, la version FastFlowLM et la mémoire disponible."
                )
            try:
                with self._request("/api/ps", timeout=1) as response:
                    models = json.load(response).get("models", [])
                if any(model.get("name") == self.tag for model in models):
                    return
            except (OSError, ValueError, URLError):
                pass
            time.sleep(0.2)
        raise RuntimeError(
            "Chargement NPU trop long ou modèle non chargé. Réessaie avec Qwen 0,8B."
        )

    def describe(self, images, prompt=DEFAULT_PROMPT, max_tokens=100, cancel=None):
        if not images:
            raise ValueError("Aucune image à analyser.")
        if cancel is not None and cancel.is_set():
            return ""
        buffer = io.BytesIO()
        images[-1].convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        request_id = uuid.uuid4().hex
        payload = {
            "model": self.tag,
            "stream": True,
            "think": False,
            "temperature": 0,
            "max_tokens": max_tokens,
            "request_id": request_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + encoded},
                        },
                    ],
                }
            ],
        }
        finished = threading.Event()

        def watch_cancel():
            while not finished.wait(0.1):
                if cancel is not None and cancel.is_set():
                    try:
                        with self._request("/api/cancel", {"request_id": request_id}, timeout=2):
                            pass
                    except (OSError, URLError):
                        pass
                    # Retry until the request is registered or finishes.

        watcher = threading.Thread(target=watch_cancel, daemon=True)
        watcher.start()
        parts = []
        try:
            with self._request("/v1/chat/completions", payload) as response:
                for raw_line in response:
                    if cancel is not None and cancel.is_set():
                        return ""
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if "error" in event:
                        raise RuntimeError(
                            "Le moteur NPU a refusé l’analyse. Vérifie le modèle et la mémoire."
                        )
                    for choice in event.get("choices", []):
                        parts.append(choice.get("delta", {}).get("content") or "")
            return "".join(parts).strip() if cancel is None or not cancel.is_set() else ""
        except (HTTPError, URLError, OSError) as error:
            if cancel is not None and cancel.is_set():
                return ""
            raise RuntimeError(
                "Le moteur NPU ne répond pas ou a refusé l’image. "
                "Essaie une résolution plus petite ou recharge le modèle."
            ) from error
        finally:
            finished.set()
            watcher.join(timeout=3)

    def synchronize(self):
        pass

    def close(self):
        with self._close_lock:
            self.closed = True
            self._stop(self.command_process)
            self._stop(self.process)
            if self.reader is not None:
                self.reader.join(timeout=2)
            if self.process is not None:
                self.process.stdout.close()
            self.process = self.reader = None
            atexit.unregister(self.close)


class LocalRedirectHandler(HTTPRedirectHandler):
    """Ne jamais rediriger les images vers un autre endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise URLError("Redirection du moteur local refusée.")
