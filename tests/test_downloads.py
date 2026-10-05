import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from downloads import model_files, prepare_model
from vision import MODEL_BY_KEY


class DownloadTests(unittest.TestCase):
    def test_shard_index_selects_only_active_weights_and_runtime_files(self):
        files = [
            "config.json",
            "hf_moondream.py",
            "model.safetensors.index.json",
            "model-00001.safetensors",
            "modelv2-00001.safetensors",
            "model_fp8.pt",
            "trainer_state.json",
            "README.md",
            "onnx/config.json",
        ]
        selected = model_files(files, {"weights": "modelv2-00001.safetensors"})
        self.assertEqual(
            selected,
            [
                "config.json",
                "hf_moondream.py",
                "model.safetensors.index.json",
                "modelv2-00001.safetensors",
            ],
        )
        with self.assertRaises(ValueError):
            model_files(files, {"weights": "../outside.safetensors"})

    def test_offline_preparation_makes_no_hub_import_or_request(self):
        progress = Mock()
        prepare_model(MODEL_BY_KEY["smol"], "cache", True, progress)
        self.assertEqual(progress.call_args.args[0]["stage"], "cache")
        prepare_model(MODEL_BY_KEY["smol"], "cache", False, None)

    def test_cached_files_skip_download_and_missing_files_have_correct_total(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index.json"
            index.write_text(json.dumps({"weight_map": {"w": "modelv2-00001.safetensors"}}))
            spec = MODEL_BY_KEY["moondream3"]

            class HubError(Exception):
                pass

            info = Mock()
            info.side_effect = [
                SimpleNamespace(
                    siblings=[
                        SimpleNamespace(rfilename="config.json", size=5),
                        SimpleNamespace(rfilename="model.safetensors.index.json", size=50),
                        SimpleNamespace(rfilename="modelv2-00001.safetensors", size=1234),
                        SimpleNamespace(rfilename="model-00001.safetensors", size=1234),
                    ]
                ),
                SimpleNamespace(siblings=[SimpleNamespace(rfilename="tokenizer.json", size=90)]),
            ]
            snapshot = Mock()
            cached = Mock(
                side_effect=lambda repo, name, **kwargs: (
                    None if name.endswith(".safetensors") else "cached"
                )
            )
            download = Mock(return_value=str(index))
            imports = {
                "httpx": SimpleNamespace(TransportError=HubError),
                "huggingface_hub": SimpleNamespace(
                    HfApi=lambda: SimpleNamespace(model_info=info),
                    hf_hub_download=download,
                    snapshot_download=snapshot,
                    try_to_load_from_cache=cached,
                ),
                "huggingface_hub.errors": SimpleNamespace(
                    HfHubHTTPError=HubError, OfflineModeIsEnabled=HubError
                ),
            }
            events = []
            with (
                patch.dict("sys.modules", imports),
                patch("downloads.progress_class", return_value="bar"),
            ):
                prepare_model(spec, directory, False, events.append)
            self.assertEqual(snapshot.call_count, 1)
            options = snapshot.call_args.kwargs
            self.assertEqual(options["revision"], spec.revision)
            self.assertNotIn("model-00001.safetensors", options["allow_patterns"])
            self.assertEqual(
                next(event["total"] for event in events if event["stage"] == "download"), 1234
            )
            self.assertEqual(info.call_args_list[1].kwargs["revision"], spec.tokenizer_revision)
            self.assertTrue(any("déjà en cache" in event.get("message", "") for event in events))


if __name__ == "__main__":
    unittest.main()
