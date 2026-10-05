"""La fenêtre Tkinter de FastVideo, sans dépendance graphique supplémentaire."""

import tkinter as tk
from tkinter import ttk

BG = "#0b1317"
PANEL = "#132127"
FIELD = "#1b2c33"
LINE = "#2b4048"
TEXT = "#edf4f1"
MUTED = "#91a8ac"
ACCENT = "#b9ee7b"
TEAL = "#64d8c3"


def label(parent, text=None, variable=None, size=10, color=TEXT, bold=False, **kwargs):
    return tk.Label(
        parent,
        text=text,
        textvariable=variable,
        bg=parent.cget("bg"),
        fg=color,
        font=("Segoe UI", size, "bold" if bold else "normal"),
        **kwargs,
    )


def card(parent, **kwargs):
    return tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1, **kwargs)


def build_interface(app):
    root = app.root
    root.title("FastVideo · Vision locale")
    root.geometry("1360x900")
    root.minsize(1140, 780)
    root.configure(bg=BG)
    root.protocol("WM_DELETE_WINDOW", app.close)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "TButton",
        font=("Segoe UI", 10),
        padding=(12, 10),
        background=FIELD,
        foreground=TEXT,
        borderwidth=0,
        focusthickness=2,
        focuscolor=TEAL,
    )
    style.map(
        "TButton",
        background=[("active", "#2b4148"), ("disabled", PANEL)],
        foreground=[("disabled", "#526a72")],
    )
    style.configure(
        "Primary.TButton", background=ACCENT, foreground=BG, font=("Segoe UI", 10, "bold")
    )
    style.map(
        "Primary.TButton",
        background=[("active", "#d2ffa1"), ("disabled", FIELD)],
        foreground=[("disabled", "#526a72")],
    )
    style.configure(
        "TCombobox",
        fieldbackground=FIELD,
        background=LINE,
        foreground=TEXT,
        arrowcolor=TEAL,
        padding=8,
        borderwidth=0,
        bordercolor=LINE,
        lightcolor=LINE,
        darkcolor=LINE,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", FIELD)],
        foreground=[("readonly", TEXT)],
        selectbackground=[("readonly", FIELD)],
        selectforeground=[("readonly", TEXT)],
    )
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", "#36564b")
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
    style.configure(
        "TSpinbox",
        fieldbackground=FIELD,
        foreground=TEXT,
        background=LINE,
        arrowcolor=TEAL,
        padding=7,
        borderwidth=0,
        bordercolor=LINE,
        lightcolor=LINE,
        darkcolor=LINE,
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=FIELD,
        background=TEAL,
        borderwidth=0,
        lightcolor=TEAL,
        darkcolor=TEAL,
        bordercolor=FIELD,
    )
    style.configure(
        "Vertical.TScrollbar",
        background=LINE,
        troughcolor=PANEL,
        arrowcolor=MUTED,
        borderwidth=0,
        bordercolor=PANEL,
        lightcolor=LINE,
        darkcolor=LINE,
    )

    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=24, pady=(19, 16))
    mark = tk.Canvas(header, width=42, height=42, bg=BG, highlightthickness=0)
    mark.pack(side="left", padx=(0, 12))
    mark.create_rectangle(3, 3, 39, 39, outline=ACCENT, width=2)
    mark.create_oval(13, 13, 29, 29, outline=ACCENT, width=2)
    mark.create_oval(18, 18, 24, 24, fill=TEAL, outline="")
    title = tk.Frame(header, bg=BG)
    title.pack(side="left")
    label(title, "FastVideo", size=23, bold=True).pack(anchor="w")
    label(title, "Une webcam. Douze modèles. Ton PC.", color=MUTED, size=9).pack(anchor="w")
    label(header, "●  INFÉRENCE LOCALE", color=ACCENT, size=10, bold=True).pack(side="right")

    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=24)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)
    sidebar = card(body, width=292)
    sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
    sidebar.pack_propagate(False)
    settings_canvas = tk.Canvas(sidebar, bg=PANEL, highlightthickness=0, width=276)
    settings_scroll = ttk.Scrollbar(sidebar, command=settings_canvas.yview)
    settings_scroll.pack(side="right", fill="y")
    settings_canvas.pack(side="left", fill="both", expand=True)
    settings_canvas.configure(yscrollcommand=settings_scroll.set)
    settings_outer = tk.Frame(settings_canvas, bg=PANEL)
    settings_window = settings_canvas.create_window(
        0, 0, window=settings_outer, anchor="nw", width=258
    )
    settings = tk.Frame(settings_outer, bg=PANEL)
    settings.pack(fill="both", expand=True, padx=16, pady=18)
    settings_outer.bind(
        "<Configure>",
        lambda event: settings_canvas.configure(scrollregion=settings_canvas.bbox("all")),
    )
    settings_canvas.bind(
        "<Configure>",
        lambda event: settings_canvas.itemconfigure(settings_window, width=event.width),
    )

    def scroll_settings(event):
        try:
            widget = root.winfo_containing(event.x_root, event.y_root)
        except (KeyError, tk.TclError):
            # Les menus ttk sont des widgets Tcl internes ("popdown"), sans
            # objet Python. Leur défilement reste géré par le menu lui-même.
            return
        if widget is None or not str(widget).startswith(str(sidebar)):
            return
        if isinstance(widget, (tk.Text, ttk.Combobox, ttk.Spinbox)):
            return
        if settings_outer.winfo_height() <= settings_canvas.winfo_height():
            return
        steps = (
            (-1 if event.num == 4 else 1)
            if getattr(event, "num", None) in (4, 5)
            else -int(event.delta / 120)
        )
        settings_canvas.yview_scroll(steps, "units")

    root.bind_all("<MouseWheel>", scroll_settings, add=True)
    root.bind_all("<Button-4>", scroll_settings, add=True)
    root.bind_all("<Button-5>", scroll_settings, add=True)

    def section(text):
        label(settings, text, color=MUTED, size=9, bold=True).pack(anchor="w", pady=(16, 9))

    label(settings, "Poste de commande", size=15, bold=True).pack(anchor="w")
    section("01  /  MODÈLE VISION")
    app.model_input = ttk.Combobox(
        settings,
        textvariable=app.selected_model,
        state="readonly",
        values=app.model_labels,
        width=23,
    )
    app.model_input.pack(fill="x")
    app.model_input.bind("<<ComboboxSelected>>", app.change_model)
    label(
        settings, variable=app.model_hint, color=MUTED, size=9, wraplength=250, justify="left"
    ).pack(anchor="w", pady=(9, 10))
    app.rocm_toggle = ttk.Checkbutton(
        settings,
        text="Attention ROCm expérimentale",
        variable=app.rocm_experimental,
        command=app.toggle_rocm_attention,
    )
    app.rocm_toggle.pack(anchor="w")
    label(
        settings, variable=app.rocm_hint, color=MUTED, size=8, wraplength=240, justify="left"
    ).pack(anchor="w", pady=(5, 10))
    app.load_button = ttk.Button(settings, text="Charger le modèle", command=app.load_model)
    app.load_button.pack(fill="x")
    section("02  /  CAPTURE & CADENCE")
    capture = tk.Frame(settings, bg=PANEL)
    capture.pack(fill="x", pady=(0, 9))
    capture.columnconfigure(0, weight=1)
    label(capture, "Résolution webcam", color=MUTED, size=8).grid(
        row=0, column=0, sticky="w", pady=(0, 5)
    )
    label(capture, "FPS cible", color=MUTED, size=8).grid(
        row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 5)
    )
    ttk.Combobox(
        capture,
        state="readonly",
        textvariable=app.capture_resolution,
        values=["640x480", "1280x720", "1920x1080", "2560x1440", "3840x2160"],
        width=14,
    ).grid(row=1, column=0, sticky="ew")
    ttk.Spinbox(capture, from_=1, to=60, textvariable=app.capture_fps, width=5).grid(
        row=1, column=1, padx=(8, 0), sticky="ew"
    )
    capture_options = tk.Frame(settings, bg=PANEL)
    capture_options.pack(fill="x", pady=(0, 9))
    label(capture_options, "Format USB", color=MUTED, size=8).pack(side="left", padx=(0, 8))
    ttk.Combobox(
        capture_options,
        state="readonly",
        textvariable=app.capture_format,
        values=["auto", "mjpg", "yuy2"],
        width=7,
    ).pack(side="left")
    ttk.Button(settings, text="Appliquer à la webcam", command=app.apply_camera_settings).pack(
        fill="x", pady=(0, 12)
    )
    inputs = tk.Frame(settings, bg=PANEL)
    inputs.pack(fill="x")
    for column, (name, variable, start, end, step) in enumerate(
        (
            ("Caméra", app.camera_index, 0, 9, 1),
            ("Intervalle / s", app.interval, 0.5, 30, 0.5),
            ("Images", app.frame_count, 1, 3, 1),
        )
    ):
        cell = tk.Frame(inputs, bg=PANEL)
        cell.grid(row=0, column=column, sticky="ew", padx=(0, 8 if column < 2 else 0))
        inputs.columnconfigure(column, weight=1)
        label(cell, name, color=MUTED, size=8).pack(anchor="w", pady=(0, 5))
        field = ttk.Spinbox(
            cell, from_=start, to=end, increment=step, width=5, textvariable=variable
        )
        field.pack(fill="x")
        if column == 0:
            app.camera_input = field
    app.camera_button = ttk.Button(settings, text="Ouvrir la webcam", command=app.toggle_camera)
    app.camera_button.pack(fill="x", pady=(12, 8))
    app.analysis_button = ttk.Button(
        settings,
        text="▶  Analyser en direct",
        style="Primary.TButton",
        command=app.toggle_analysis,
        state="disabled",
    )
    app.analysis_button.pack(fill="x")
    app.single_button = ttk.Button(
        settings, text="Analyser une image", command=app.analyze_once, state="disabled"
    )
    app.single_button.pack(fill="x", pady=(8, 0))
    section("03  /  CONSIGNE")
    app.preset = ttk.Combobox(settings, state="readonly", values=list(app.presets), width=23)
    app.preset.current(0)
    app.preset.pack(fill="x", pady=(0, 8))
    app.preset.bind("<<ComboboxSelected>>", app.apply_preset)
    app.prompt = tk.Text(
        settings,
        height=4,
        wrap="word",
        font=("Segoe UI", 10),
        bg=FIELD,
        fg=TEXT,
        insertbackground=TEAL,
        relief="flat",
        padx=10,
        pady=10,
        highlightthickness=1,
        highlightbackground=LINE,
        highlightcolor=TEAL,
    )
    app.prompt.insert("1.0", app.presets["Scène & objets"])
    app.prompt.pack(fill="x")
    label(settings, "Longueur maximale / tokens", color=MUTED, size=9).pack(
        anchor="w", pady=(12, 5)
    )
    ttk.Spinbox(settings, from_=1, to=512, increment=10, textvariable=app.max_tokens, width=8).pack(
        anchor="w"
    )
    label(
        settings,
        "Les poids sont téléchargés au premier usage.\nLes images restent sur ce PC.",
        color=MUTED,
        size=9,
        justify="left",
    ).pack(side="bottom", anchor="w", pady=(18, 0))

    workspace = tk.Frame(body, bg=BG)
    workspace.grid(row=0, column=1, sticky="nsew")
    metrics = tk.Frame(workspace, bg=BG)
    metrics.pack(fill="x", pady=(0, 14))
    for column, (name, variable, note) in enumerate(
        (
            ("CAMÉRA", app.fps_text, "images / seconde"),
            ("DERNIÈRE ANALYSE", app.latency_text, "durée de l’inférence"),
            ("RYTHME OBSERVÉ", app.rate_text, "réponses / minute"),
            ("MOTEUR", app.backend_text, "calcul sur ce PC"),
        )
    ):
        box = card(metrics)
        box.grid(row=0, column=column, sticky="nsew", padx=(0, 10 if column < 3 else 0))
        metrics.columnconfigure(column, weight=1, uniform="metric")
        label(box, name, color=MUTED, size=8, bold=True).pack(anchor="w", padx=13, pady=(10, 2))
        label(
            box, variable=variable, size=19, bold=True, color=ACCENT if column == 1 else TEXT
        ).pack(anchor="w", padx=13)
        label(box, note, color=MUTED, size=8).pack(anchor="w", padx=13, pady=(1, 10))

    view = tk.Frame(workspace, bg=BG)
    view.pack(fill="both", expand=True)
    view.columnconfigure(0, weight=3, uniform="view")
    view.columnconfigure(1, weight=2, uniform="view")
    view.rowconfigure(0, weight=1)
    camera = card(view)
    camera.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
    top = tk.Frame(camera, bg=PANEL)
    top.pack(fill="x", padx=14, pady=12)
    label(top, "Flux webcam", size=12, bold=True).pack(side="left")
    ttk.Button(top, text="Agrandir", command=app.expand_preview).pack(side="right", padx=(10, 0))
    label(top, variable=app.camera_state, color=TEAL, size=9).pack(side="right")
    app.preview = tk.Label(
        camera,
        text="La scène commence ici.\n\nOuvre ta webcam pour voir le flux.",
        bg="#0d181c",
        fg=MUTED,
        font=("Segoe UI", 12),
        width=1,
        height=1,
    )
    app.preview.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    camera_footer = tk.Frame(camera, bg=PANEL)
    camera_footer.pack(fill="x", padx=14, pady=(0, 12))
    label(
        camera_footer,
        variable=app.camera_details,
        color=MUTED,
        size=8,
        wraplength=350,
        justify="left",
    ).pack(side="left", fill="x", expand=True)
    style.configure("TCheckbutton", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
    style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("active", TEXT)])
    ttk.Checkbutton(camera_footer, text="Miroir", variable=app.mirror).pack(side="right")
    response = card(view)
    response.grid(row=0, column=1, sticky="nsew")
    top = tk.Frame(response, bg=PANEL)
    top.pack(fill="x", padx=14, pady=(12, 5))
    label(top, "Ce que voit l’IA", size=12, bold=True).pack(side="left")
    app.copy_button = ttk.Button(top, text="Copier", command=app.copy_result, state="disabled")
    app.copy_button.pack(side="right")
    label(response, variable=app.answer_time, color=MUTED, size=9).pack(
        anchor="w", padx=14, pady=(0, 10)
    )
    app.output = tk.Text(
        response,
        wrap="word",
        font=("Segoe UI", 12),
        bg=PANEL,
        fg=TEXT,
        relief="flat",
        padx=14,
        pady=10,
        state="disabled",
        width=1,
        height=4,
    )
    app.output.pack(fill="both", expand=True)
    label(response, variable=app.stats, color=TEAL, size=9, wraplength=280, justify="left").pack(
        anchor="w", padx=14, pady=14
    )
    app.set_output(
        "La description apparaîtra ici.\n\nChoisis un modèle, ouvre la webcam et lance l’analyse."
    )

    journal = card(workspace)
    journal.pack(fill="x", pady=(14, 0))
    top = tk.Frame(journal, bg=PANEL)
    top.pack(fill="x", padx=14, pady=(10, 6))
    label(top, "Journal d’activité", size=11, bold=True).pack(side="left")
    ttk.Button(top, text="Effacer", command=app.clear_log).pack(side="right", padx=(8, 0))
    ttk.Button(top, text="Exporter…", command=app.export_log).pack(side="right")
    progress = tk.Frame(journal, bg=PANEL)
    progress.pack(fill="x", padx=14, pady=(0, 6))
    app.progress_bar = ttk.Progressbar(progress, mode="determinate", maximum=100)
    app.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 12))
    label(progress, variable=app.download_text, color=TEAL, size=9).pack(side="right")
    lines = tk.Frame(journal, bg=PANEL)
    lines.pack(fill="both", padx=14, pady=(0, 12))
    app.log_view = tk.Text(
        lines,
        height=5,
        wrap="word",
        font=("Consolas", 9),
        bg="#0d181c",
        fg=MUTED,
        relief="flat",
        padx=9,
        pady=8,
        state="disabled",
    )
    app.log_view.pack(side="left", fill="both", expand=True)
    scroll = ttk.Scrollbar(lines, command=app.log_view.yview)
    scroll.pack(side="right", fill="y")
    app.log_view.configure(yscrollcommand=scroll.set)
    app.log_view.tag_configure("error", foreground="#ff9c91")
    app.log_view.tag_configure("success", foreground=ACCENT)

    footer = tk.Frame(root, bg=BG)
    footer.pack(fill="x", padx=24, pady=12)
    label(footer, variable=app.status, color=TEAL, size=9, anchor="w", wraplength=1000).pack(
        side="left"
    )
    label(footer, "FASTVIDEO / LOCAL VISION", color=MUTED, size=8).pack(side="right")
