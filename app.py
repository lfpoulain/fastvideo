"""Une webcam, une fenêtre Python, et un petit modèle vision exécuté sur le PC."""

import argparse
import logging
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import filedialog

import cv2
from PIL import Image, ImageOps, ImageTk

from vision import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    MODEL_BY_KEY,
    MODELS,
    LocalVision,
    configure_rocm_attention,
    describe_timed,
)

DEFAULT_LOG_FILE = Path(__file__).resolve().parent / "logs" / "fastvideo.log"
CAPTURE_RESOLUTIONS = ("640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160")


def format_bytes(value):
    for unit in ("o", "Kio", "Mio", "Gio"):
        if value < 1024 or unit == "Gio":
            return f"{value:.1f} {unit}"
        value /= 1024


def make_logger(filename):
    logger = logging.getLogger(f"fastvideo.{time.time_ns()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    error = None
    try:
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(filename, maxBytes=2 * 1024**2, backupCount=2, encoding="utf-8")
        file.setFormatter(formatter)
        logger.addHandler(file)
    except OSError as failure:
        error = f"Fichier de logs indisponible · {failure}"
    return logger, error


class Camera:
    """La capture tourne indépendamment du modèle et de l'interface."""

    def __init__(self, index, events, resolution="1920x1080", fps=25, pixel_format="auto"):
        self.index, self.events = index, events
        self.width, self.height = map(int, resolution.split("x"))
        self.requested_fps, self.pixel_format = fps, pixel_format
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.samples = deque(maxlen=3)
        self.frame_times = deque(maxlen=120)
        self.thread = threading.Thread(target=self._capture, daemon=True)

    def _capture(self):
        camera = None
        try:
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            camera = cv2.VideoCapture(self.index, backend)
            if not camera.isOpened():
                raise RuntimeError(
                    f"Webcam {self.index} introuvable ou déjà utilisée par une autre app."
                )
            # MJPG limite la bande passante USB en HD ; le pilote peut refuser.
            codec = (
                "MJPG"
                if self.pixel_format == "auto" and self.width >= 1280
                else self.pixel_format.upper()
            )
            if codec != "AUTO":
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            camera.set(cv2.CAP_PROP_FPS, self.requested_fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_sample, failures, ready = 0.0, 0, False
            while not self.stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        raise RuntimeError("Le flux webcam a été interrompu.")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                height, width = frame.shape[:2]
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                sample = None
                if now - last_sample >= 1:
                    sample = image.copy()
                    sample.thumbnail((640, 480))
                    last_sample = now
                with self.lock:
                    self.latest = image
                    self.frame_times.append(now)
                    if sample is not None:
                        self.samples.append((now, sample))
                if not ready:
                    ready = True
                    reported_fps = camera.get(cv2.CAP_PROP_FPS)
                    fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
                    actual_codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
                    if not actual_codec.isascii() or not actual_codec.isprintable():
                        actual_codec = "non annoncé"
                    self.events.put(
                        (
                            "camera_ready",
                            self,
                            {
                                "width": width,
                                "height": height,
                                "fps": reported_fps,
                                "codec": actual_codec,
                                "requested": (self.width, self.height, self.requested_fps),
                            },
                        )
                    )
        except Exception as error:
            if not self.stop.is_set():
                self.events.put(("camera_error", self, str(error)))
        finally:
            if camera is not None:
                camera.release()

    def fps(self):
        with self.lock:
            recent = [at for at in self.frame_times if time.monotonic() - at < 2]
        if len(recent) < 2 or time.monotonic() - recent[-1] > 1:
            return 0.0
        return (len(recent) - 1) / max(0.001, recent[-1] - recent[0])

    def snapshot(self, count):
        with self.lock:
            if self.latest is None:
                return []
            now = time.monotonic()
            recent = [image for at, image in self.samples if 0.5 <= now - at < 4]
            images = [image.copy() for image in recent[-(count - 1) :]] if count > 1 else []
            images.append(self.latest.copy())
        for image in images:
            image.thumbnail((640, 480))
        return images


class App:
    presets = {
        "Scène & objets": DEFAULT_PROMPT,
        "Réponse courte": "Décris la scène en français en une seule phrase courte.",
        "Changements visibles": "Décris en français les changements visibles entre les images.",
        "Texte visible": "Lis le texte visible dans l’image, sans inventer les mots illisibles.",
    }

    def __init__(self, root, args):
        from interface import build_interface

        self.root, self.args = root, args
        self.events = queue.Queue()
        self.camera = self.engine = self.worker = None
        self.cancel = threading.Event()
        self.closing = self.analyzing = False
        self.session = 0
        self.next_analysis = 0.0
        self.rendered_frame = self.rendered_size = None
        self.completed = deque(maxlen=30)
        self.log_lines = deque(maxlen=400)
        self.model_labels = [model.label for model in MODELS]
        self.selected_model = tk.StringVar(value=MODEL_BY_KEY[args.model].label)
        self.camera_index = tk.IntVar(value=args.camera)
        self.interval = tk.DoubleVar(value=args.interval)
        self.frame_count = tk.IntVar(value=args.frames)
        self.max_tokens = tk.IntVar(value=args.max_tokens)
        self.capture_resolution = tk.StringVar(
            value=getattr(args, "capture_resolution", "1920x1080")
        )
        self.capture_fps = tk.IntVar(value=getattr(args, "capture_fps", 25))
        self.capture_format = tk.StringVar(value=getattr(args, "capture_format", "auto"))
        self.mirror = tk.BooleanVar(value=True)
        self.camera_details = tk.StringVar(value="Résolution et FPS réglables · aperçu en miroir")
        self.resume_after_camera = self.restarting_camera = False
        self.preview_window = self.large_preview = None
        self.large_rendered = None
        self.status = tk.StringVar(value="Prêt · choisis un modèle et ouvre ta webcam.")
        self.stats = tk.StringVar(value="Aucune analyse pour le moment.")
        self.fps_text = tk.StringVar(value="—")
        self.latency_text = tk.StringVar(value="—")
        self.rate_text = tk.StringVar(value="—")
        self.backend_text = tk.StringVar(
            value="AUTO" if args.device == "auto" else args.device.upper()
        )
        self.camera_state = tk.StringVar(value="●  ARRÊTÉE")
        self.answer_time = tk.StringVar(value="En attente d’une scène")
        self.download_text = tk.StringVar(value="Aucun téléchargement en cours")
        self.model_hint = tk.StringVar()
        self.logger, log_error = make_logger(getattr(args, "log_file", DEFAULT_LOG_FILE))
        self.progress_mode = "determinate"
        self.last_metrics = 0.0
        self.last_controls = None
        build_interface(self)
        self.update_model_hint()
        self.record(
            "FastVideo prêt · " + ("mode hors ligne" if args.offline else "modèles à la demande")
        )
        if log_error:
            self.record(log_error, "error")
        else:
            self.record(f"Logs persistants · {getattr(args, 'log_file', DEFAULT_LOG_FILE)}")
        root.after(30, self.tick)

    def set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def record(self, message, level="info"):
        self.logger.log(logging.ERROR if level == "error" else logging.INFO, message)
        line = f"{time.strftime('%H:%M:%S')}  {message}"
        expired = (
            self.log_lines[0].count("\n") + 1 if len(self.log_lines) == self.log_lines.maxlen else 0
        )
        self.log_lines.append(line)
        self.log_view.configure(state="normal")
        follow = self.log_view.yview()[1] >= 0.98
        self.log_view.insert("end", line + "\n", level)
        if expired:
            self.log_view.delete("1.0", f"{expired + 1}.0")
        line_count = int(self.log_view.index("end-1c").split(".")[0])
        if line_count > 800:
            self.log_view.delete("1.0", f"{line_count - 800}.0")
        if follow:
            self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def clear_log(self):
        self.log_lines.clear()
        self.log_view.configure(state="normal")
        self.log_view.delete("1.0", "end")
        self.log_view.configure(state="disabled")

    def export_log(self):
        filename = filedialog.asksaveasfilename(
            parent=self.root,
            title="Exporter le journal",
            defaultextension=".log",
            initialfile=f"fastvideo-{time.strftime('%Y%m%d-%H%M%S')}.log",
            filetypes=[("Journal", "*.log"), ("Texte", "*.txt")],
        )
        if filename:
            try:
                Path(filename).write_text("\n".join(self.log_lines) + "\n", encoding="utf-8")
                self.record("Journal exporté · " + filename, "success")
            except OSError as error:
                self.record(f"Export impossible · {error}", "error")
                self.status.set("Impossible d’exporter le journal. Voir l’erreur ci-dessus.")

    def copy_result(self):
        text = self.output.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.record("Description copiée dans le presse-papiers.")

    def apply_preset(self, event=None):
        self.prompt.delete("1.0", "end")
        self.prompt.insert("1.0", self.presets[self.preset.get()])
        self.record("Consigne sélectionnée · " + self.preset.get())

    def update_model_hint(self):
        spec = MODEL_BY_KEY[self.model_key()]
        frames = (
            "Dernière image uniquement" if spec.max_frames == 1 else "Jusqu’à 3 images récentes"
        )
        self.model_hint.set(frames + " · poids téléchargés une fois.")

    def toggle_camera(self):
        if self.camera is not None:
            self.resume_after_camera = self.restarting_camera = False
            self.pause_analysis()
            stopped_camera = self.camera
            stopped_camera.stop.set()
            self.camera = None
            self.camera_button.configure(text="Ouvrir la webcam", state="disabled")

            def camera_released():
                if self.closing:
                    return
                if stopped_camera.thread.is_alive():
                    self.root.after(50, camera_released)
                else:
                    self.camera_button.configure(state="normal")

            self.root.after(50, camera_released)
            self.camera_input.configure(state="normal")
            self.preview.configure(
                image="", text="Webcam arrêtée.\nTu peux la rouvrir quand tu veux."
            )
            self.rendered_frame = self.rendered_size = None
            self.camera_state.set("●  ARRÊTÉE")
            self.fps_text.set("—")
            self.status.set("Webcam arrêtée.")
            self.record("Webcam fermée.")
            self.sync_controls()
            return
        try:
            index = self.camera_index.get()
            if not 0 <= index <= 9:
                raise ValueError()
        except (tk.TclError, ValueError):
            self.status.set("Choisis un numéro de webcam entre 0 et 9.")
            return
        try:
            resolution, fps, pixel_format = self.capture_settings()
        except (tk.TclError, ValueError):
            self.status.set("Résolution invalide ou FPS hors plage (1 à 60).")
            return
        self.camera = Camera(index, self.events, resolution, fps, pixel_format)
        self.camera.thread.start()
        self.camera_button.configure(text="Fermer la webcam")
        self.camera_input.configure(state="disabled")
        self.camera_state.set("●  OUVERTURE")
        self.status.set("Ouverture de la webcam…")
        self.record(
            f"Ouverture de la webcam {index} · {resolution} · {fps} FPS demandés · format {pixel_format}"
        )

    def capture_settings(self):
        resolution, fps, pixel_format = (
            self.capture_resolution.get(),
            self.capture_fps.get(),
            self.capture_format.get(),
        )
        if (
            resolution not in CAPTURE_RESOLUTIONS
            or not 1 <= fps <= 60
            or pixel_format not in ("auto", "mjpg", "yuy2")
        ):
            raise ValueError("Réglages webcam invalides")
        return resolution, fps, pixel_format

    def apply_camera_settings(self):
        try:
            self.capture_settings()
        except (tk.TclError, ValueError):
            self.status.set("Résolution invalide ou FPS hors plage (1 à 60).")
            return
        if self.camera is None:
            self.status.set("Les réglages seront utilisés à l’ouverture de la webcam.")
            return
        if self.restarting_camera:
            return
        resume = self.analyzing
        self.toggle_camera()
        self.resume_after_camera = resume
        self.restarting_camera = True
        self.status.set("Application des réglages webcam…")
        self.record("Modification des réglages · réouverture de la webcam, modèle conservé.")

        def reopen():
            if self.closing or not self.restarting_camera:
                return
            if self.camera_button.instate(["disabled"]):
                self.root.after(50, reopen)
            else:
                self.toggle_camera()
                if self.camera is None:
                    self.restarting_camera = self.resume_after_camera = False

        self.root.after(60, reopen)

    def expand_preview(self):
        if self.preview_window is not None:
            self.preview_window.lift()
            return
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("FastVideo · Aperçu webcam")
        self.preview_window.geometry("1280x760")
        self.preview_window.configure(bg="#0d181c")
        self.large_preview = tk.Label(
            self.preview_window,
            bg="#0d181c",
            fg="#91a8ac",
            text="Ouvre la webcam pour afficher le flux.",
        )
        self.large_preview.pack(fill="both", expand=True)
        self.large_rendered = None
        self.preview_window.protocol("WM_DELETE_WINDOW", self.close_preview)
        self.preview_window.bind("<Escape>", lambda event: self.close_preview())
        self.preview_window.bind(
            "<F11>",
            lambda event: self.preview_window.attributes(
                "-fullscreen", not self.preview_window.attributes("-fullscreen")
            ),
        )

    def close_preview(self):
        if self.preview_window is not None:
            self.preview_window.destroy()
            self.preview_window = self.large_preview = None
            self.large_rendered = None

    def pause_analysis(self):
        was_active = self.analyzing
        self.analyzing = False
        self.session += 1
        self.cancel.set()
        if was_active:
            self.record("Analyse en pause.")
        self.sync_controls()

    def change_model(self, event=None):
        self.session += 1
        self.cancel.set()
        self.next_analysis = 0
        self.completed.clear()
        self.latency_text.set("—")
        self.rate_text.set("—")
        self.stats.set(f"{self.selected_model.get()} · sélectionné")
        self.update_model_hint()
        self.set_progress_mode("determinate")
        self.progress_bar["value"] = 0
        self.download_text.set("Modèle sélectionné · en attente")
        self.record("Modèle sélectionné · " + self.selected_model.get())
        self.status.set(
            "Changement de modèle…" if self.analyzing else "Charge le modèle ou lance une analyse."
        )
        self.sync_controls()

    def model_key(self):
        return MODELS[self.model_input.current()].key

    def toggle_analysis(self):
        if self.analyzing:
            self.pause_analysis()
            self.status.set("Analyse en pause. La webcam reste active.")
        elif self.camera is not None:
            self.analyzing = True
            self.completed.clear()
            self.rate_text.set("—")
            self.session += 1
            self.next_analysis = 0
            self.status.set("Préparation de l’analyse locale…")
            self.record("Analyse en direct démarrée · " + self.selected_model.get())
            self.sync_controls()

    def load_model(self):
        self.start_worker([])

    def analyze_once(self):
        if self.camera is not None:
            images = self.camera.snapshot(self.selected_frames())
            if images:
                self.pause_analysis()
                self.start_worker(images)
            else:
                self.status.set("La webcam n’a pas encore fourni d’image.")

    def selected_frames(self):
        try:
            count = min(3, max(1, self.frame_count.get()))
        except tk.TclError:
            count = self.args.frames
        return min(count, MODEL_BY_KEY[self.model_key()].max_frames)

    def sync_controls(self):
        busy = self.worker is not None and self.worker.is_alive()
        ready = self.camera is not None and self.camera.latest is not None
        current = (busy, ready, self.analyzing)
        if current == self.last_controls:
            return
        self.last_controls = current
        self.load_button.configure(state="disabled" if busy else "normal")
        self.single_button.configure(state="normal" if ready and not busy else "disabled")
        self.analysis_button.configure(
            text="Ⅱ  Mettre en pause" if self.analyzing else "▶  Analyser en direct",
            state="normal" if ready else "disabled",
        )

    def start_worker(self, images):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            tokens = min(512, max(1, self.max_tokens.get()))
        except tk.TclError:
            self.status.set("Choisis une longueur de réponse entre 1 et 512 tokens.")
            return
        self.cancel = threading.Event()
        self.worker = threading.Thread(
            target=self.analyze,
            args=(
                images,
                self.prompt.get("1.0", "end"),
                self.model_key(),
                self.session,
                self.cancel,
                tokens,
                self.camera,
            ),
            daemon=True,
        )
        self.worker.start()
        self.sync_controls()

    def analyze(self, images, prompt, key, session, cancel, max_tokens, camera):
        def progress(data):
            self.events.put(("progress", session, dict(data, key=key)))

        try:
            if self.engine is not None and self.engine.spec.key != key:
                self.events.put(("log", session, "Libération du modèle précédent…"))
                self.engine.close()
                self.engine = None
            if self.engine is None:
                self.engine = LocalVision(
                    key, self.args.device, self.args.offline, progress=progress
                )
            else:
                progress(
                    {
                        "stage": "ready",
                        "message": f"Modèle déjà chargé · {MODEL_BY_KEY[key].label}",
                        "backend": self.engine.backend,
                    }
                )
            if cancel.is_set() or not images:
                return
            # Le chargement peut durer : utiliser le flux actuel après sa fin.
            if camera is not None and camera is self.camera:
                images = camera.snapshot(len(images))
            if not images:
                return
            self.events.put(("status", session, "Analyse en cours · " + self.engine.backend))
            text, elapsed = describe_timed(self.engine, images, prompt, max_tokens, cancel)
            if not cancel.is_set():
                if not text:
                    raise RuntimeError("Le modèle n’a pas renvoyé de description. Réessaie.")
                memory = None
                if self.engine.device == "cuda":
                    memory = self.engine.torch.cuda.memory_allocated() / 1024**3
                self.events.put(
                    ("result", session, (text, elapsed, len(images), self.engine.backend, memory))
                )
        except Exception as error:
            self.logger.exception("Échec du modèle %s", key)
            self.events.put(
                (
                    "failure_log",
                    session,
                    (key, f"{MODEL_BY_KEY[key].label} · {type(error).__name__} : {error}"),
                )
            )
            if not cancel.is_set():
                self.events.put(("error", session, f"{type(error).__name__} : {error}"))
        finally:
            if self.closing and self.engine is not None:
                self.engine.close()
                self.engine = None

    def show_progress(self, data):
        stage = data["stage"]
        if message := data.get("message"):
            self.record(message, "success" if stage in ("ready", "cache") else "info")
        if stage == "diagnostic":
            return
        if data["key"] != self.model_key():
            return
        if stage == "download":
            self.set_progress_mode("determinate")
            total, current = data.get("total", 0), data.get("current", 0)
            percent = min(100, current / total * 100) if total else 0
            self.progress_bar["value"] = percent
            self.download_text.set(
                f"{percent:.0f}% · {format_bytes(current)} / {format_bytes(total)} · {format_bytes(data.get('rate', 0))}/s"
            )
            self.status.set("Téléchargement des fichiers du modèle…")
        elif stage in ("load", "check"):
            self.set_progress_mode("indeterminate")
            self.download_text.set(
                "Chargement en mémoire…" if stage == "load" else "Vérification du cache…"
            )
            self.status.set(data["message"])
        else:
            self.set_progress_mode("determinate")
            self.progress_bar["value"] = 100 if stage == "ready" else 0
            self.download_text.set(
                "Modèle prêt" if stage == "ready" else data["message"].split(" · ")[0]
            )
            if stage == "ready":
                backend = data.get("backend", "CPU")
                self.backend_text.set(backend.split(" · ")[0])
                self.stats.set(backend + (" · " + data["dtype"] if "dtype" in data else ""))
                self.status.set(
                    "Modèle prêt. "
                    + (
                        "Analyse en direct active."
                        if self.analyzing
                        else "Tu peux lancer une analyse."
                    )
                )

    def set_progress_mode(self, mode):
        if mode != self.progress_mode:
            self.progress_bar.stop()
            self.progress_bar.configure(mode=mode)
            self.progress_mode = mode
            if mode == "indeterminate":
                self.progress_bar.start(18)

    def handle_event(self, kind, origin, data):
        if kind.startswith("camera_"):
            if origin is not self.camera:
                return
            if kind == "camera_error":
                self.restarting_camera = self.resume_after_camera = False
                self.toggle_camera()
                self.status.set(data)
                self.record(data, "error")
            else:
                self.status.set("Webcam active. Lance l’analyse quand tu veux.")
                self.camera_state.set("●  EN DIRECT")
                fps = data["fps"]
                reported = f"{fps:g} FPS annoncés" if fps > 0 else "FPS non annoncés par le pilote"
                self.camera_details.set(
                    f"{data['width']}×{data['height']} · {reported} · {data['codec']}"
                )
                self.record("Webcam active · " + self.camera_details.get(), "success")
                width, height, requested_fps = data["requested"]
                if (width, height) != (data["width"], data["height"]) or (
                    fps > 0 and abs(fps - requested_fps) > 0.5
                ):
                    self.record(
                        f"Le pilote a choisi un autre mode que {width}×{height} à {requested_fps} FPS. Les FPS mesurés figurent dans le compteur caméra."
                    )
                self.restarting_camera = False
                if self.resume_after_camera:
                    self.resume_after_camera = False
                    self.toggle_analysis()
            return
        if kind == "progress":
            self.show_progress(data)
            return
        if kind in ("log", "failure_log"):
            if kind == "failure_log":
                key, message = data
                self.record(message, "error")
                if key == self.model_key():
                    self.set_progress_mode("determinate")
                    self.progress_bar["value"] = 0
                    self.download_text.set("Erreur · consulter le journal")
            else:
                self.record(data)
            return
        if origin != self.session:
            return
        if kind == "result":
            text, elapsed, count, backend, memory = data
            self.set_output(text)
            self.copy_button.configure(state="normal")
            self.answer_time.set("Dernière réponse · " + time.strftime("%H:%M:%S"))
            self.latency_text.set(f"{elapsed:.2f} s")
            self.completed.append(time.monotonic())
            if len(self.completed) > 1:
                rate = (
                    60
                    * (len(self.completed) - 1)
                    / max(0.001, self.completed[-1] - self.completed[0])
                )
                self.rate_text.set(f"{rate:.1f}")
            detail = f"{count} image(s) · {backend}"
            if memory is not None:
                detail += f"\nMémoire PyTorch · {memory:.2f} Gio"
            self.stats.set(detail)
            self.record(
                f"{self.selected_model.get()} · réponse en {elapsed:.2f} s · {count} image(s)",
                "success",
            )
            if len(self.completed) == 1 and elapsed >= 10 and backend.startswith("ROCm"):
                self.record(
                    "Première analyse lente · initialisation des kernels possible. Compare les suivantes ; pour réduire la latence, essaie LFM 450M ou 1,6B et une réponse courte."
                )
            self.status.set(
                "Analyse active · les prochaines images seront les plus récentes."
                if self.analyzing
                else "Analyse terminée."
            )
        elif kind == "error":
            self.pause_analysis()
            self.status.set("Analyse arrêtée après une erreur. Tu peux réessayer.")
            self.set_output(data)
            self.copy_button.configure(state="disabled")
        else:
            self.status.set(data)

    def tick(self):
        if self.closing:
            return
        while True:
            try:
                self.handle_event(*self.events.get_nowait())
            except queue.Empty:
                break
        if self.camera is not None:
            with self.camera.lock:
                latest = self.camera.latest
            if latest is not None:
                size = (max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height()))
                render_state = (size, self.mirror.get())
                if latest is not self.rendered_frame or render_state != self.rendered_size:
                    scaled = ImageOps.contain(latest, size)
                    self.photo = ImageTk.PhotoImage(
                        ImageOps.mirror(scaled) if self.mirror.get() else scaled
                    )
                    self.preview.configure(image=self.photo, text="")
                    self.rendered_frame, self.rendered_size = latest, render_state
                if self.large_preview is not None:
                    large_size = (
                        max(1, self.large_preview.winfo_width()),
                        max(1, self.large_preview.winfo_height()),
                    )
                    large_state = (latest, large_size, self.mirror.get())
                    if (
                        self.large_rendered is None
                        or latest is not self.large_rendered[0]
                        or large_state[1:] != self.large_rendered[1:]
                    ):
                        scaled = ImageOps.contain(latest, large_size)
                        self.large_photo = ImageTk.PhotoImage(
                            ImageOps.mirror(scaled) if self.mirror.get() else scaled
                        )
                        self.large_preview.configure(image=self.large_photo, text="")
                        self.large_rendered = large_state
            if (
                self.analyzing
                and (self.worker is None or not self.worker.is_alive())
                and time.monotonic() >= self.next_analysis
            ):
                images = self.camera.snapshot(self.selected_frames())
                if images:
                    try:
                        interval = min(30.0, max(0.5, self.interval.get()))
                    except tk.TclError:
                        interval = self.args.interval
                    self.next_analysis = time.monotonic() + interval
                    self.start_worker(images)
        if time.monotonic() - self.last_metrics >= 0.5:
            self.last_metrics = time.monotonic()
            if self.camera is not None:
                self.fps_text.set(f"{self.camera.fps():.1f}")
        self.sync_controls()
        if self.camera is None and self.large_preview is not None:
            self.large_preview.configure(image="", text="Webcam arrêtée")
            self.large_rendered = None
        self.root.after(30, self.tick)

    def close(self):
        self.closing = True
        self.pause_analysis()
        self.record("Fermeture de FastVideo.")
        if self.camera is not None:
            self.camera.stop.set()
            self.camera.thread.join(timeout=1)
        if self.engine is not None and (self.worker is None or not self.worker.is_alive()):
            self.engine.close()
            self.engine = None
        self.root.destroy()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Numéro de la webcam (0 par défaut)")
    parser.add_argument(
        "--capture-resolution",
        choices=CAPTURE_RESOLUTIONS,
        default="1920x1080",
        help="Résolution demandée à la webcam",
    )
    parser.add_argument(
        "--capture-fps", type=int, default=25, help="FPS demandés à la webcam, de 1 à 60"
    )
    parser.add_argument(
        "--capture-format",
        choices=["auto", "mjpg", "yuy2"],
        default="auto",
        help="Format USB ; auto essaie MJPG en HD",
    )
    parser.add_argument("--model", choices=list(MODEL_BY_KEY), default=DEFAULT_MODEL)
    parser.add_argument(
        "--list-models", action="store_true", help="Afficher les modèles disponibles"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
    parser.add_argument(
        "--rocm-experimental-attention",
        action="store_true",
        help="Autoriser les kernels SDPA ROCm expérimentaux au lancement (gain non garanti)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2,
        help="Intervalle minimum entre les analyses, en secondes",
    )
    parser.add_argument(
        "--frames",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Images récentes par analyse ; FastVLM et Moondream utilisent la dernière",
    )
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument(
        "--offline", action="store_true", help="Utiliser uniquement le modèle déjà téléchargé"
    )
    parser.add_argument(
        "--image", type=Path, help="Analyser une image locale en ligne de commande, sans webcam"
    )
    parser.add_argument(
        "--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Fichier du journal de l’interface"
    )
    args = parser.parse_args()
    if args.list_models:
        for spec in MODELS:
            print(f"{spec.key:12}  {spec.label:22}  {spec.repository}")
        return
    if (
        not 0 <= args.camera <= 9
        or not 0.5 <= args.interval <= 30
        or not 1 <= args.max_tokens <= 512
        or not 1 <= args.capture_fps <= 60
    ):
        parser.error(
            "Webcam : 0 à 9 ; capture-fps : 1 à 60 ; intervalle : 0,5 à 30 s ; max-tokens : 1 à 512."
        )
    configure_rocm_attention(args.rocm_experimental_attention)
    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
            image.thumbnail((640, 480))
        engine = LocalVision(args.model, args.device, args.offline)
        try:
            for message in engine.diagnostics:
                print(message)
            description, elapsed = describe_timed(engine, [image], DEFAULT_PROMPT, args.max_tokens)
            print(description)
            print(f"{engine.backend} · {elapsed:.2f} s")
        finally:
            engine.close()
    else:
        root = tk.Tk()
        App(root, args)
        root.mainloop()


if __name__ == "__main__":
    main()
