import os
import subprocess
import sys
import unittest
from pathlib import Path

from vision import MODELS

APP = Path(__file__).resolve().parents[1] / "app.py"


class CliTests(unittest.TestCase):
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
        ]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
