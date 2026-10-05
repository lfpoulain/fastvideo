"""Préparer uniquement les fichiers utiles, avec une progression Hugging Face réelle."""

import json
import time
from pathlib import PurePosixPath


def model_files(files, weight_map=None):
    """Éviter les variantes ONNX, les poids alternatifs et les fichiers d'entraînement."""
    selected = {
        name
        for name in files
        if "/" not in name
        and name.endswith((".json", ".py", ".txt", ".model", ".jinja", ".tiktoken"))
        and name != "trainer_state.json"
    }
    weights = set(weight_map.values()) if weight_map is not None else {"model.safetensors"}
    for name in weights:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not name.endswith(".safetensors"):
            raise ValueError(f"Nom de poids inattendu : {name}")
        if name not in files:
            raise ValueError(f"Poids absent du dépôt : {name}")
    return sorted(selected | weights)


def progress_class(callback, total):
    from tqdm.auto import tqdm

    class DownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            self.last_report = 0.0
            self.downloaded = 0
            self.started = time.monotonic()
            kwargs.pop("name", None)
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            # La barre de fichiers sert à thread_map ; seule celle en octets
            # fournit la progression de téléchargement à l'interface.
            if getattr(self, "unit", None) != "B":
                return
            now = time.monotonic()
            if now - self.last_report >= 0.2:
                self.last_report = now
                callback(
                    {
                        "stage": "download",
                        "current": self.n,
                        "total": total,
                        "rate": self.downloaded / max(0.001, now - self.started),
                    }
                )

        def update(self, n=1):
            self.downloaded += n or 0
            return super().update(n)

    return DownloadProgress


def prepare_model(spec, cache_dir, offline, callback):
    """Le CLI sans callback conserve le chargement direct d'origine."""
    if callback is None:
        return
    if offline:
        callback({"stage": "cache", "message": "Mode hors ligne · lecture du cache local."})
        return

    import httpx
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    repositories = [(spec.repository, spec.revision, False)]
    if spec.tokenizer_repository:
        repositories.append((spec.tokenizer_repository, spec.tokenizer_revision, True))
    for repository, revision, tokenizer_only in repositories:
        callback({"stage": "check", "message": f"Vérification des fichiers · {repository}"})
        try:
            info = HfApi().model_info(repository, revision=revision, files_metadata=True)
        except (httpx.TransportError, HfHubHTTPError, OfflineModeIsEnabled):
            callback({"stage": "cache", "message": "Hub indisponible · tentative depuis le cache."})
            return
        files = {file.rfilename: file for file in info.siblings}
        if tokenizer_only:
            selected = ["tokenizer.json"]
        else:
            weight_map = None
            if "model.safetensors.index.json" in files:
                index = hf_hub_download(
                    repository,
                    "model.safetensors.index.json",
                    revision=revision,
                    cache_dir=cache_dir,
                )
                with open(index, encoding="utf-8") as source:
                    weight_map = json.load(source)["weight_map"]
            selected = model_files(files, weight_map)
        missing = [
            name
            for name in selected
            if not isinstance(
                try_to_load_from_cache(repository, name, revision=revision, cache_dir=cache_dir),
                str,
            )
        ]
        if not missing:
            callback({"stage": "cache", "message": f"Fichiers déjà en cache · {repository}"})
            continue
        total = sum(files[name].size or 0 for name in missing)
        callback(
            {
                "stage": "download",
                "message": f"Téléchargement · {repository} · {len(missing)} fichier(s)",
                "current": 0,
                "total": total,
                "rate": 0,
            }
        )
        snapshot_download(
            repository,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=selected,
            max_workers=4,
            tqdm_class=progress_class(callback, total),
        )
        callback({"stage": "cache", "message": f"Téléchargement terminé · {repository}"})
