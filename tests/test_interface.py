import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import App


class InterfaceTests(unittest.TestCase):
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
