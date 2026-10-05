import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from app import main
from vision import MODELS

APP = Path(__file__).resolve().parents[1] / "app.py"


class CliTests(unittest.TestCase):
    def test_attention_option_is_set_before_opening_gui_and_preserves_existing_env(self):
        variable = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
        cases = [((), "0", "0"), ((), "1", "1"), (("--rocm-experimental-attention",), "0", "1")]
        for args, initial, expected in cases:
            with (
                self.subTest(args=args, initial=initial),
                patch.dict(os.environ, {variable: initial}),
                patch.object(sys, "argv", [str(APP), *args]),
                patch("app.tk.Tk") as root,
            ):

                def opened(*_args):
                    self.assertEqual(os.environ[variable], expected)

                with patch("app.App", side_effect=opened) as dashboard:
                    main()
                dashboard.assert_called_once()
                root.return_value.mainloop.assert_called_once()

    def run_cli(self, *args):
        environment = dict(os.environ, PYTHONUTF8="1")
        return subprocess.run(
            [sys.executable, str(APP), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            env=environment,
        )

    def test_list_models_without_opening_a_window_or_loading_weights(self):
        result = self.run_cli("--list-models")
        self.assertEqual(result.returncode, 0, result.stderr)
        for spec in MODELS:
            self.assertIn(spec.key, result.stdout)
            self.assertIn(spec.repository, result.stdout)

    def test_invalid_options_fail_before_capture_or_inference(self):
        for args in [
            ("--camera", "-1"),
            ("--interval", "nan"),
            ("--max-tokens", "0"),
            ("--frames", "4"),
            ("--model", "unknown"),
            ("--capture-fps", "0"),
            ("--capture-fps", "61"),
            ("--capture-resolution", "bogus"),
        ]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
