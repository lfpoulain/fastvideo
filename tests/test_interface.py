import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace

from app import App


class InterfaceTests(unittest.TestCase):
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
