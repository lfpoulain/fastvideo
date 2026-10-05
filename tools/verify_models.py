"""Vérifier la génération des modèles sur une image synthétique, hors webcam."""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import MODEL_BY_KEY, LocalVision, describe_timed  # noqa: E402


def make_image():
    image = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 80, 260, 290), fill="red")
    draw.ellipse((340, 80, 550, 290), fill="blue")
    draw.text((180, 380), "RED SQUARE + BLUE CIRCLE", fill="black", font_size=24)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_BY_KEY), default=list(MODEL_BY_KEY)
    )
    parser.add_argument("--precision", choices=["auto", "fp16"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-report.json"))
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    image = make_image()
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key in args.models:
        engine = None
        record = {
            "model": key,
            "repository": MODEL_BY_KEY[key].repository,
            "revision": MODEL_BY_KEY[key].revision,
        }
        print(f"{key}: chargement…", flush=True)
        try:
            start = time.perf_counter()
            engine = LocalVision(key, offline=args.offline)
            record["load_seconds"] = round(time.perf_counter() - start, 3)
            if args.precision == "fp16":
                if engine.device != "cuda":
                    raise RuntimeError("Le test FP16 nécessite un GPU CUDA ou ROCm.")
                engine.dtype = engine.torch.float16
                engine.model.to(dtype=engine.dtype)
            record["backend"] = engine.backend
            record["dtype"] = str(engine.dtype)
            if engine.device == "cuda":
                record["allocated_gib"] = round(engine.torch.cuda.memory_allocated() / 1024**3, 3)
            prompt = "Décris les formes et leurs couleurs en français, en une phrase."
            for run in ("cold", "warm"):
                answer, elapsed = describe_timed(engine, [image], prompt, 48)
                if not answer.strip():
                    raise RuntimeError("Réponse vide.")
                record[f"{run}_seconds"] = round(elapsed, 3)
                record[f"{run}_answer"] = answer
            if engine.spec.max_frames > 1:
                answer, _ = describe_timed(engine, [image, image], prompt, 20)
                if not answer.strip():
                    raise RuntimeError("Réponse multi-image vide.")
            cancelled = threading.Event()
            cancelled.set()
            if engine.describe([image], cancel=cancelled) != "":
                raise RuntimeError("L'annulation avant génération a échoué.")
            record["ok"] = True
            print(f"{key}: OK · {record['warm_seconds']:.2f} s", flush=True)
        except Exception as error:
            record["ok"] = False
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"{key}: ERREUR · {record['error']}", flush=True)
        finally:
            if engine is not None:
                engine.close()
            results.append(record)
            args.output.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    print(f"Rapport : {args.output.resolve()}")
    return 0 if all(record["ok"] for record in results) else 1


if __name__ == "__main__":
    sys.exit(main())
