import base64
import io
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from npu import NPU_MODELS, NpuVision, find_flm
from vision import describe_timed

FAKE_FLM = Path(__file__).with_name("fake_flm.py")


class NpuTests(unittest.TestCase):
    def fake_engine(self, key="qwen-0.8b", **kwargs):
        spawn = NpuVision._spawn

        def fake_spawn(engine, arguments):
            return spawn(engine, [str(FAKE_FLM), *arguments])

        with patch.object(NpuVision, "_spawn", fake_spawn):
            return NpuVision(key, executable=sys.executable, **kwargs)

    def test_unsupported_model_and_strict_offline_fail_before_starting_anything(self):
        with patch("npu.find_flm") as locate:
            with self.assertRaisesRegex(ValueError, "pas de version NPU"):
                NpuVision("minicpm")
            with self.assertRaisesRegex(ValueError, "offline strict"):
                NpuVision("qwen-2b", offline=True)
            locate.assert_not_called()

    def test_missing_executable_has_an_installation_message(self):
        with (
            patch("npu.shutil.which", return_value=None),
            patch.dict("npu.os.environ", {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "Installer FastFlowLM"):
                find_flm()

    def test_lifecycle_image_payload_streaming_and_no_gpu_dependency(self):
        messages = []
        engine = self.fake_engine("qwen-2b", progress=messages.append)
        process = engine.process
        try:
            self.assertEqual(engine.device, "npu")
            self.assertEqual(engine.spec.max_frames, 1)
            self.assertFalse(hasattr(engine, "torch"))
            text, elapsed = describe_timed(
                engine,
                [Image.new("RGB", (20, 20), "blue"), Image.new("RGB", (320, 180), "red")],
                "Décris cette image.",
                40,
            )
            self.assertEqual(text, "Un carré rouge.")
            self.assertGreater(elapsed, 0)
            with engine._request("/test/payload", timeout=2) as response:
                payload = json.load(response)
            self.assertEqual(payload["model"], NPU_MODELS["qwen-2b"])
            self.assertFalse(payload["think"])
            self.assertEqual(payload["max_tokens"], 40)
            content = payload["messages"][0]["content"]
            self.assertEqual(len(content), 2)
            image_url = content[1]["image_url"]["url"]
            with Image.open(io.BytesIO(base64.b64decode(image_url.split(",", 1)[1]))) as image:
                self.assertEqual(image.size, (320, 180))
                self.assertGreater(image.getpixel((0, 0))[0], 240)
            journal = "\n".join(message["message"] for message in messages)
            self.assertIn("2 Mio/s", journal)
            self.assertIn("Modèle prêt", journal)
            self.assertNotIn("data:image", journal)
            self.assertNotIn("Décris cette image", journal)
        finally:
            engine.close()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)
        engine.close()

    def test_cancel_during_prefill_reaches_the_matching_request_and_suppresses_reply(self):
        engine = self.fake_engine()
        cancel = threading.Event()
        timer = threading.Timer(0.3, cancel.set)
        started = time.monotonic()
        try:
            timer.start()
            self.assertEqual(
                engine.describe([Image.new("RGB", (16, 16))], "WAIT", cancel=cancel), ""
            )
            self.assertLess(time.monotonic() - started, 3)
            cancel.clear()
            self.assertEqual(
                engine.describe([Image.new("RGB", (16, 16))], cancel=cancel), "Un carré rouge."
            )
        finally:
            timer.cancel()
            timer.join()
            engine.close()

    def test_server_error_and_redirect_do_not_leak_images_or_hang(self):
        engine = self.fake_engine()
        try:
            for prompt in ("FAIL", "REDIRECT"):
                with (
                    self.subTest(prompt=prompt),
                    self.assertRaisesRegex(RuntimeError, "moteur NPU"),
                ):
                    engine.describe([Image.new("RGB", (16, 16))], prompt)
        finally:
            engine.close()

    def test_loading_failure_terminates_its_private_process(self):
        process = None

        def failed(engine, timeout):
            nonlocal process
            process = engine.process
            raise RuntimeError("Chargement simulé en échec")

        with patch.object(NpuVision, "_wait_ready", failed):
            with self.assertRaisesRegex(RuntimeError, "en échec"):
                self.fake_engine()
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdout.closed)


if __name__ == "__main__":
    unittest.main()
