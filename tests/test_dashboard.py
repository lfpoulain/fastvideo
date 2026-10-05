import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import App
from vision import MODEL_BY_KEY


class DashboardTests(unittest.TestCase):
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
        app.camera = SimpleNamespace(snapshot=Mock(return_value=["fresh frame"]))
        engine = SimpleNamespace(spec=MODEL_BY_KEY["smol"], device="cpu", backend="CPU")
        with (
            patch("app.LocalVision", return_value=engine),
            patch("app.describe_timed", return_value=("fresh description", 0.2)) as describe,
        ):
            cancel = threading.Event()
            app.analyze(["old frame"], "Describe", "smol", 4, cancel, 32, app.camera)
        describe.assert_called_once_with(engine, ["fresh frame"], "Describe", 32, cancel)
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


if __name__ == "__main__":
    unittest.main()
