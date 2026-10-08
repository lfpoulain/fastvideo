import gc
import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import DEVICE_LABELS, App
from vision import MODEL_BY_KEY


class InterfaceTests(unittest.TestCase):
    def tearDown(self):
        # Collecter les anciennes fenêtres sur le thread Tk, avant qu'un worker
        # du test suivant ne déclenche le GC et les destructeurs des StringVar.
        gc.collect()

    def test_npu_filters_models_and_resolution_changes_keep_engine_and_full_size_preview(self):
        from PIL import Image

        from app import Camera

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Affichage Tk indisponible : {error}")
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model="minicpm",
                camera=0,
                interval=2,
                frames=3,
                max_tokens=40,
                device="auto",
                offline=False,
                log_file=Path(directory) / "test.log",
            )
            app = App(root, args)
            engine = SimpleNamespace(
                spec=MODEL_BY_KEY["qwen-0.8b"], close=Mock(), backend="NPU AMD", device="npu"
            )
            try:
                app.selected_device.set(DEVICE_LABELS["npu"])
                app.device_input.event_generate("<<ComboboxSelected>>")
                self.assertEqual(
                    [spec.key for spec in app.available_models], ["qwen-0.8b", "qwen-2b", "qwen-4b"]
                )
                self.assertEqual(app.model_key(), "qwen-0.8b")
                self.assertEqual(app.selected_frames(), 1)
                self.assertTrue(app.rocm_toggle.instate(["disabled"]))
                self.assertFalse(app.npu_install_button.instate(["disabled"]))
                self.assertFalse(app.npu_path_button.instate(["disabled"]))
                with patch("app.NpuVision", return_value=engine), patch("app.LocalVision") as gpu:
                    app.load_button.invoke()
                    app.worker.join(timeout=3)
                gpu.assert_not_called()
                self.assertIs(app.engine, engine)
                self.assertFalse(app.runtime_locked)
                app.camera = Camera(0, app.events)
                app.camera.latest = Image.new("RGB", (1920, 1080))
                app.analysis_resolution.set("320x240")
                app.analysis_resolution_input.event_generate("<<ComboboxSelected>>")
                with patch("app.describe_timed", return_value=("Scène simulée.", 0.1)) as describe:
                    app.analyze_once()
                    app.worker.join(timeout=3)
                images = describe.call_args.args[1]
                self.assertEqual([image.size for image in images], [(320, 180)])
                self.assertEqual(app.camera.latest.size, (1920, 1080))
                self.assertIs(app.engine, engine)
                engine.close.assert_not_called()
                app.selected_device.set(DEVICE_LABELS["cpu"])
                app.device_input.event_generate("<<ComboboxSelected>>")
                self.assertEqual(len(app.available_models), 12)
                self.assertEqual(app.model_key(), "qwen-0.8b")
                self.assertFalse(app.rocm_toggle.instate(["disabled"]))
                self.assertTrue(app.npu_path_button.instate(["disabled"]))
            finally:
                app.camera = None  # Capture simulée : le thread n'a pas été démarré.
                app.close()

    def test_npu_file_picker_replaces_loaded_engine_without_a_terminal_command(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Affichage Tk indisponible : {error}")
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model="qwen-0.8b",
                camera=0,
                interval=2,
                frames=1,
                max_tokens=40,
                device="npu",
                offline=False,
                log_file=Path(directory) / "test.log",
            )
            app = App(root, args)
            previous = SimpleNamespace(close=Mock())
            app.engine = previous
            try:
                chosen = str(Path(directory) / "Custom install" / "flm.exe")
                with patch("app.filedialog.askopenfilename", return_value=""):
                    app.npu_path_button.invoke()
                previous.close.assert_not_called()
                with patch("app.filedialog.askopenfilename", return_value=chosen):
                    app.npu_path_button.invoke()
                self.assertEqual(args.flm_path, Path(chosen))
                previous.close.assert_called_once()
                self.assertIsNone(app.engine)
                self.assertIn(chosen, "\n".join(app.log_lines))
                engine = SimpleNamespace(spec=MODEL_BY_KEY["qwen-0.8b"], close=Mock())
                with patch("app.NpuVision", return_value=engine) as load:
                    app.load_button.invoke()
                    app.worker.join(timeout=3)
                self.assertEqual(load.call_args.kwargs["executable"], Path(chosen))
            finally:
                app.close()

    def test_rocm_checkbox_applies_before_loading_and_locks_for_the_session(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Affichage Tk indisponible : {error}")
        root.withdraw()
        variable = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {variable: "0"}):
            args = SimpleNamespace(
                model="smol",
                camera=0,
                interval=2,
                frames=1,
                max_tokens=100,
                device="auto",
                offline=True,
                log_file=Path(directory) / "test.log",
            )
            app = App(root, args)
            engine = SimpleNamespace(close=Mock())
            observed = []

            def loaded(*_args, **_kwargs):
                observed.append(os.environ[variable])
                return engine

            try:
                self.assertFalse(app.rocm_experimental.get())
                app.rocm_toggle.invoke()
                self.assertEqual(os.environ[variable], "1")
                app.rocm_toggle.invoke()
                self.assertEqual(os.environ[variable], "0")
                app.rocm_toggle.invoke()
                with patch("app.LocalVision", side_effect=loaded):
                    app.load_button.invoke()
                    app.worker.join(timeout=3)
                self.assertFalse(app.worker.is_alive())
                self.assertEqual(observed, ["1"])
                self.assertIs(app.engine, engine)
                self.assertTrue(app.rocm_toggle.instate(["disabled"]))
                self.assertIn("Relance", app.rocm_hint.get())
                app.rocm_toggle.invoke()
                self.assertTrue(app.rocm_experimental.get())
                self.assertEqual(os.environ[variable], "1")
            finally:
                app.close()

    def test_mouse_wheel_over_native_combobox_popup_does_not_crash_or_scroll_sidebar(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Affichage Tk indisponible : {error}")
        root.withdraw()
        failures = []
        root.report_callback_exception = lambda *error: failures.append(error)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                model="smol",
                camera=0,
                interval=2,
                frames=1,
                max_tokens=100,
                device="auto",
                offline=True,
                log_file=Path(directory) / "test.log",
            )
            app = App(root, args)
            try:
                root.geometry("1140x780+30+30")
                root.attributes("-topmost", True)
                root.deiconify()
                root.update()
                root.tk.call("ttk::combobox::Post", str(app.model_input))
                root.update()
                popdown = root.tk.call("ttk::combobox::PopdownWindow", str(app.model_input))
                listbox = str(popdown) + ".f.l"
                x = int(root.tk.call("winfo", "rootx", listbox)) + 10
                y = int(root.tk.call("winfo", "rooty", listbox)) + 10
                # Reproduire exactement le lookup Python qui échouait sur popdown.
                with self.assertRaises(KeyError):
                    root.winfo_containing(x, y)
                canvas = app.model_input.master.master.master
                before = canvas.yview()
                root.event_generate("<MouseWheel>", delta=-120, rootx=x, rooty=y)
                root.update()
                self.assertEqual(failures, [], failures)
                self.assertEqual(canvas.yview(), before)
                root.tk.call("ttk::combobox::Unpost", str(app.model_input))
                root.update()
                # La molette continue de faire défiler le panneau sur ses contrôles.
                x = app.load_button.winfo_rootx() + 10
                y = app.load_button.winfo_rooty() + 10
                root.event_generate("<MouseWheel>", delta=-120, rootx=x, rooty=y)
                root.update()
                self.assertGreater(canvas.yview()[0], before[0])
                self.assertEqual(failures, [], failures)
            finally:
                root.tk.call("ttk::combobox::Unpost", str(app.model_input))
                app.close()


if __name__ == "__main__":
    unittest.main()
