import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def launchers(self):
        if os.name == "nt":
            powershell = shutil.which("powershell")
            if powershell:
                yield (
                    "powershell",
                    [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"],
                )
            bash = Path(r"C:\Program Files\Git\bin\bash.exe")
            if bash.is_file():
                yield "bash", [str(bash)]
        elif shutil.which("bash"):
            yield "bash", [shutil.which("bash")]

    def run_launcher(self, directory, name, command, exit_code=0, missing_python=False):
        for script in ("setup.ps1", "setup.sh"):
            shutil.copyfile(ROOT / script, directory / script)
        tools = directory / "tools"
        tools.mkdir(exist_ok=True)
        (tools / "bootstrap.py").write_text(
            "import sys\n"
            "print('child stdout', flush=True)\n"
            "print('child stderr', file=sys.stderr, flush=True)\n"
            f"sys.exit({exit_code})\n",
            encoding="utf-8",
        )
        python = str(directory / "missing-python") if missing_python else sys.executable
        environment = dict(os.environ, FASTVIDEO_PYTHON=python.replace("\\", "/"), PYTHONUTF8="1")
        arguments = (
            [str(directory / "setup.ps1"), "-Python", python]
            if name == "powershell"
            else [str(directory / "setup.sh")]
        )
        return subprocess.run(
            [*command, *arguments],
            cwd=directory.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def test_stdout_stderr_and_child_exit_code_are_saved_even_with_spaces_in_path(self):
        for name, command in self.launchers():
            for code in (0, 7):
                with self.subTest(launcher=name, code=code), tempfile.TemporaryDirectory() as temp:
                    directory = Path(temp) / "project with spaces"
                    directory.mkdir()
                    result = self.run_launcher(directory, name, command, code)
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                    logs = list((directory / "logs").glob("startup-*.log"))
                    self.assertEqual(len(logs), 1)
                    content = logs[0].read_text(encoding="utf-8-sig", errors="replace")
                    self.assertIn("child stdout", content)
                    self.assertIn("child stderr", content)
                    self.assertIn(f"Exit code: {code}", content)
                    self.assertIn("Startup log:", result.stdout)

    def test_missing_python_is_logged_before_bootstrap_can_start(self):
        for name, command in self.launchers():
            with self.subTest(launcher=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                result = self.run_launcher(directory, name, command, missing_python=True)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                logs = list((directory / "logs").glob("startup-*.log"))
                self.assertEqual(len(logs), 1)
                content = logs[0].read_text(encoding="utf-8-sig", errors="replace")
                self.assertIn("missing-python", content)
                self.assertIn("Exit code: 1", content)
                self.assertNotIn("child stdout", content)


if __name__ == "__main__":
    unittest.main()
