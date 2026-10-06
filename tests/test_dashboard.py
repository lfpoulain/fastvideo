import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from app import App
from vision import MODEL_BY_KEY


class DashboardTests(unittest.TestCase):
    def test_kernel_diagnostic_is_logged_without_changing_download_progress(self):
        app = App.__new__(App)
        app.record = Mock()
        app.set_progress_mode = Mock()
        app.show_progress({"stage": "diagnostic", "message": "HIP · gfx1150"})
        app.record.assert_called_once_with("HIP · gfx1150", "info")
        app.set_progress_mode.assert_not_called()

    def test_old_results_and_errors_cannot_replace_current_session(self):
        app = App.__new__(App)
        app.session = 4
        app.set_output = Mock()
        app.pause_analysis = Mock()
        app.handle_event("result", 3, ("old", 1, 1, "CPU", None))
        app.handle_event("error", 3, "old failure")
        app.set_output.assert_not_called()
        app.pause_analysis.assert_not_called()

    def test_frame_is_recaptured_after_slow_model_loading(self):
        app = App.__new__(App)
        app.events = queue.Queue()
        app.engine = None
        app.closing = False
        app.args = SimpleNamespace(device="cpu", offline=True)
        app.logger = Mock()
        fresh = Image.new("RGB", (640, 360))
        app.camera = SimpleNamespace(snapshot=Mock(return_value=[fresh]))
        engine = SimpleNamespace(spec=MODEL_BY_KEY["smol"], device="cpu", backend="CPU")
        with (
            patch("app.LocalVision", return_value=engine),
            patch("app.describe_timed", return_value=("fresh description", 0.2)) as describe,
        ):
            cancel = threading.Event()
            app.analyze(["old frame"], "Describe", "smol", 4, cancel, 32, app.camera)
        describe.assert_called_once_with(engine, [fresh], "Describe", 32, cancel)
        app.camera.snapshot.assert_called_once_with(1, "640x480")
        results = list(app.events.queue)
        self.assertTrue(any(kind == "result" and origin == 4 for kind, origin, _ in results))

    def test_failure_from_previous_model_is_logged_without_changing_progress(self):
        app = App.__new__(App)
        app.record = Mock()
        app.model_key = Mock(return_value="qwen-4b")
        app.set_progress_mode = Mock()
        app.handle_event("failure_log", 3, ("smol", "old load failed"))
        app.record.assert_called_once_with("old load failed", "error")
        app.set_progress_mode.assert_not_called()

    def test_pause_during_loading_prevents_generation_and_keeps_loaded_model(self):
        app = App.__new__(App)
        app.events = queue.Queue()
        app.engine = None
        app.closing = False
        app.args = SimpleNamespace(device="cpu", offline=True)
        app.logger = Mock()
        cancel = threading.Event()
        engine = SimpleNamespace(spec=MODEL_BY_KEY["smol"], device="cpu", backend="CPU")

        def loaded(*args, **kwargs):
            cancel.set()
            kwargs["progress"]({"stage": "ready", "message": "ready"})
            return engine

        with patch("app.LocalVision", side_effect=loaded), patch("app.describe_timed") as describe:
            app.analyze(["image"], "Describe", "smol", 4, cancel, 32, None)
        describe.assert_not_called()
        self.assertIs(app.engine, engine)
        self.assertEqual(app.events.get_nowait()[0], "progress")

    def test_switching_runtime_releases_old_engine_even_for_the_same_model(self):
        app = App.__new__(App)
        app.events = queue.Queue()
        app.closing = False
        app.args = SimpleNamespace(device="auto", offline=False)
        app.logger = Mock()
        old = SimpleNamespace(spec=MODEL_BY_KEY["qwen-2b"], close=Mock())
        app.engine, app.engine_request = old, "auto"
        new = SimpleNamespace(spec=MODEL_BY_KEY["qwen-2b"], backend="NPU AMD", close=Mock())
        with patch("app.NpuVision", return_value=new) as loader:
            app.analyze([], "Describe", "qwen-2b", 1, threading.Event(), 32, None, "npu")
        old.close.assert_called_once()
        loader.assert_called_once()
        self.assertIs(app.engine, new)
        self.assertEqual(app.engine_request, "npu")


if __name__ == "__main__":
    unittest.main()
