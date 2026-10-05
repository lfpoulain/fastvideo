"""Une webcam, une fenêtre Python, et un petit modèle vision exécuté sur le PC."""

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageOps, ImageTk

from vision import DEFAULT_MODEL, DEFAULT_PROMPT, MODEL_BY_KEY, MODELS, LocalVision, describe_timed


class Camera:
    """La capture tourne indépendamment du modèle et de l'interface."""

    def __init__(self, index, events):
        self.index, self.events = index, events
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.samples = deque(maxlen=3)
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
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            last_sample, failures = 0.0, 0
            self.events.put(("camera_ready", self, "Webcam active"))
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
                scale = min(640 / width, 480 / height, 1.0)
                if scale < 1:
                    frame = cv2.resize(
                        frame,
                        (round(width * scale), round(height * scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                now = time.monotonic()
                with self.lock:
                    self.latest = image
                    if now - last_sample >= 1:
                        sample = image.copy()
                        sample.thumbnail((640, 480))
                        self.samples.append((now, sample))
                        last_sample = now
        except Exception as error:
            if not self.stop.is_set():
                self.events.put(("camera_error", self, str(error)))
        finally:
            if camera is not None:
                camera.release()

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
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.events = queue.Queue()
        self.camera = None
        self.engine = None
        self.worker = None
        self.cancel = threading.Event()
        self.closing = False
        self.session = 0
        self.next_analysis = 0.0
        self.analyzing = False
        self.rendered_frame = None
        self.rendered_size = None
        self.selected_model = tk.StringVar(value=MODEL_BY_KEY[args.model].label)
        self.camera_index = tk.IntVar(value=args.camera)
        self.interval = tk.DoubleVar(value=args.interval)
        self.status = tk.StringVar(value="Ouvre la webcam pour commencer.")
        self.stats = tk.StringVar(value="Inférence locale · aucune clé API")

        root.title("FastVideo — webcam + IA locale")
        root.geometry("1100x720")
        root.minsize(900, 600)
        root.configure(bg="#f5f5f2")
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f5f2")
        style.configure("TLabel", background="#f5f5f2", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)

        header = ttk.Frame(root, padding=(20, 16))
        header.pack(fill="x")
        ttk.Label(header, text="FastVideo", font=("Segoe UI", 22, "bold")).pack(side="left")
        ttk.Label(header, text="WEBCAM + VISION LOCALE", foreground="#657269").pack(side="right")
        controls = ttk.Frame(root, padding=(20, 0, 20, 12))
        controls.pack(fill="x")
        ttk.Label(controls, text="Webcam").pack(side="left")
        self.camera_input = ttk.Spinbox(
            controls, from_=0, to=9, width=3, textvariable=self.camera_index
        )
        self.camera_input.pack(side="left", padx=(8, 16))
        self.camera_button = ttk.Button(
            controls, text="Ouvrir la webcam", command=self.toggle_camera
        )
        self.camera_button.pack(side="left")
        self.analysis_button = ttk.Button(
            controls, text="Analyser en direct", command=self.toggle_analysis, state="disabled"
        )
        self.analysis_button.pack(side="left", padx=8)
        ttk.Label(controls, text="Intervalle (s)").pack(side="left", padx=(16, 8))
        ttk.Spinbox(
            controls, from_=0.5, to=30, increment=0.5, width=4, textvariable=self.interval
        ).pack(side="left")

        model_row = ttk.Frame(root, padding=(20, 0, 20, 14))
        model_row.pack(fill="x")
        ttk.Label(model_row, text="Modèle").pack(side="left", padx=(0, 10))
        self.model_input = ttk.Combobox(
            model_row,
            textvariable=self.selected_model,
            state="readonly",
            width=26,
            values=[model.label for model in MODELS],
        )
        self.model_input.pack(side="left")
        self.model_input.bind("<<ComboboxSelected>>", self.change_model)
        ttk.Label(
            model_row, text="Téléchargé une fois · CUDA / ROCm / CPU", foreground="#657269"
        ).pack(side="left", padx=16)

        body = ttk.Frame(root, padding=(20, 0, 20, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.preview = tk.Label(
            body, text="Ta webcam apparaîtra ici", bg="#18251f", fg="#becbc2", font=("Segoe UI", 14)
        )
        self.preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        panel = ttk.Frame(body)
        panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(panel, text="Ce que voit le modèle", font=("Segoe UI", 16, "bold")).pack(
            anchor="w", pady=(0, 12)
        )
        self.output = tk.Text(
            panel,
            wrap="word",
            font=("Segoe UI", 12),
            bg="white",
            fg="#22352a",
            relief="flat",
            padx=14,
            pady=14,
            state="disabled",
            width=32,
        )
        self.output.pack(fill="both", expand=True)
        self.set_output(
            "L'analyse s'affichera ici.\n\nLes images sont traitées sur ton PC. Le modèle est téléchargé au premier lancement de l'analyse."
        )
        ttk.Label(panel, textvariable=self.stats, foreground="#657269", wraplength=380).pack(
            anchor="w", pady=10
        )
        footer = ttk.Frame(root, padding=20)
        footer.pack(fill="x")
        ttk.Label(footer, text="Consigne pour l’IA").pack(anchor="w")
        self.prompt = tk.Text(
            footer, height=2, wrap="word", font=("Segoe UI", 10), relief="flat", padx=8, pady=8
        )
        self.prompt.insert("1.0", DEFAULT_PROMPT)
        self.prompt.pack(fill="x", pady=(6, 10))
        ttk.Label(footer, textvariable=self.status, foreground="#41694f", wraplength=1000).pack(
            anchor="w"
        )
        root.after(30, self.tick)

    def set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    def toggle_camera(self):
        if self.camera is not None:
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
            self.analysis_button.configure(state="disabled")
            self.preview.configure(image="", text="Webcam arrêtée")
            self.status.set("Webcam arrêtée.")
            return
        try:
            index = self.camera_index.get()
            if not 0 <= index <= 9:
                raise ValueError()
        except (tk.TclError, ValueError):
            self.status.set("Choisis un numéro de webcam entre 0 et 9.")
            return
        self.camera = Camera(index, self.events)
        self.camera.thread.start()
        self.camera_button.configure(text="Fermer la webcam")
        self.camera_input.configure(state="disabled")
        self.status.set("Ouverture de la webcam…")

    def pause_analysis(self):
        self.analyzing = False
        self.session += 1
        self.cancel.set()
        self.analysis_button.configure(text="Analyser en direct")

    def change_model(self, event=None):
        self.session += 1
        self.cancel.set()
        self.next_analysis = 0
        self.stats.set(f"{self.selected_model.get()} · sélectionné")
        self.status.set(
            "Changement de modèle…"
            if self.analyzing
            else "Le modèle sera chargé à la prochaine analyse."
        )

    def model_key(self):
        return MODELS[self.model_input.current()].key

    def toggle_analysis(self):
        if self.analyzing:
            self.pause_analysis()
            self.status.set("Analyse en pause. La webcam reste active.")
        else:
            self.analyzing = True
            self.session += 1
            self.next_analysis = 0
            self.analysis_button.configure(text="Mettre en pause")
            self.status.set("Préparation de l’analyse locale…")

    def analyze(self, images, prompt, key, session, cancel):
        try:
            if self.engine is not None and self.engine.spec.key != key:
                self.engine.close()
                self.engine = None
            if self.engine is None:
                self.events.put(
                    (
                        "status",
                        session,
                        f"Chargement de {MODEL_BY_KEY[key].label}… Téléchargement au premier usage.",
                    )
                )
                self.engine = LocalVision(key, self.args.device, self.args.offline)
            if cancel.is_set():
                return
            self.events.put(("status", session, f"Analyse sur {self.engine.backend}…"))
            text, elapsed = describe_timed(
                self.engine, images, prompt, self.args.max_tokens, cancel
            )
            if not cancel.is_set():
                if not text:
                    raise RuntimeError("Le modèle n’a pas renvoyé de description. Réessaie.")
                self.events.put(
                    ("result", session, (text, elapsed, len(images), self.engine.backend))
                )
        except Exception as error:
            if not cancel.is_set():
                self.events.put(("error", session, f"{type(error).__name__} : {error}"))
        finally:
            if self.closing and self.engine is not None:
                self.engine.close()
                self.engine = None

    def tick(self):
        if self.closing:
            return
        while not self.events.empty():
            kind, origin, data = self.events.get_nowait()
            if kind.startswith("camera_"):
                if origin is not self.camera:
                    continue
                if kind == "camera_error":
                    self.toggle_camera()
                    self.status.set(data)
                else:
                    self.analysis_button.configure(state="normal")
                    self.status.set("Webcam active. Lance l’analyse quand tu veux.")
                continue
            if origin != self.session:
                continue
            if kind == "result":
                text, elapsed, count, backend = data
                self.set_output(text)
                self.stats.set(
                    f"{self.selected_model.get()}\n{backend}\n{elapsed:.2f} s · {count} image(s) · {time.strftime('%H:%M:%S')}"
                )
                self.status.set(
                    "Analyse active · la prochaine réponse utilisera les images les plus récentes."
                )
            elif kind == "error":
                self.pause_analysis()
                self.status.set("Analyse arrêtée après une erreur. Tu peux réessayer.")
                self.set_output(data)
            else:
                self.status.set(data)
        if self.camera is not None:
            with self.camera.lock:
                latest = self.camera.latest
            if latest is not None:
                size = (max(1, self.preview.winfo_width()), max(1, self.preview.winfo_height()))
                if latest is not self.rendered_frame or size != self.rendered_size:
                    self.photo = ImageTk.PhotoImage(ImageOps.contain(ImageOps.mirror(latest), size))
                    self.preview.configure(image=self.photo, text="")
                    self.rendered_frame, self.rendered_size = latest, size
            if (
                self.analyzing
                and (self.worker is None or not self.worker.is_alive())
                and time.monotonic() >= self.next_analysis
            ):
                key = self.model_key()
                count = min(self.args.frames, MODEL_BY_KEY[key].max_frames)
                images = self.camera.snapshot(count)
                if images:
                    try:
                        interval = min(30.0, max(0.5, self.interval.get()))
                    except tk.TclError:
                        interval = self.args.interval
                    self.next_analysis = time.monotonic() + interval
                    self.cancel = threading.Event()
                    self.worker = threading.Thread(
                        target=self.analyze,
                        args=(
                            images,
                            self.prompt.get("1.0", "end"),
                            key,
                            self.session,
                            self.cancel,
                        ),
                        daemon=True,
                    )
                    self.worker.start()
        self.root.after(30, self.tick)

    def close(self):
        self.closing = True
        self.pause_analysis()
        if self.camera is not None:
            self.camera.stop.set()
            self.camera.thread.join(timeout=1)
        if self.engine is not None and (self.worker is None or not self.worker.is_alive()):
            self.engine.close()
            self.engine = None
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Numéro de la webcam (0 par défaut)")
    parser.add_argument("--model", choices=list(MODEL_BY_KEY), default=DEFAULT_MODEL)
    parser.add_argument(
        "--list-models", action="store_true", help="Afficher les modèles disponibles"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "rocm", "cpu"], default="auto")
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
    args = parser.parse_args()
    if args.list_models:
        for spec in MODELS:
            print(f"{spec.key:12}  {spec.label:22}  {spec.repository}")
        return
    if (
        not 0 <= args.camera <= 9
        or not 0.5 <= args.interval <= 30
        or not 1 <= args.max_tokens <= 512
    ):
        parser.error("Webcam : 0 à 9 ; intervalle : 0,5 à 30 s ; max-tokens : 1 à 512.")
    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
            image.thumbnail((640, 480))
        engine = LocalVision(args.model, args.device, args.offline)
        try:
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
