import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import bootstrap


def installed_cuda(errors=None):
    return {
        "python": [3, 12],
        "backend": "cuda",
        "available": True,
        "torch": "2.11.0+cu128",
        "torchvision": "0.26.0+cu128",
        "errors": errors or [],
        "gpu": "Test NVIDIA",
    }


class BootstrapTests(unittest.TestCase):
    def test_hx470_and_hip_names_select_the_right_architecture(self):
        for name in ("AMD Ryzen AI 9 HX 470", "AMD Ryzen AI 9 HX470", "AMD Radeon 890M"):
            with self.subTest(name=name):
                self.assertEqual(bootstrap.amd_hardware(name), (True, "gfx1150"))
        self.assertEqual(bootstrap.amd_hardware("Name: gfx1151"), (True, "gfx1151"))
        self.assertEqual(bootstrap.amd_hardware("AMD Radeon unknown"), (True, None))
        self.assertEqual(
            bootstrap.amd_hardware("VGA controller: Advanced Micro Devices [AMD]"), (True, None)
        )
        self.assertEqual(bootstrap.amd_hardware("Intel Core Ultra"), (False, None))

    def test_auto_reuses_a_working_gpu_and_prefers_nvidia_over_an_amd_igpu(self):
        dual_gpu = {"nvidia": True, "amd": True, "arch": "gfx1150"}
        self.assertEqual(bootstrap.select_backend("auto", "auto", {}, dual_gpu), ("cuda", None))
        rocm = {"backend": "rocm", "available": True, "arch": "gfx1150"}
        self.assertEqual(
            bootstrap.select_backend("auto", "auto", rocm, dual_gpu), ("rocm", "gfx1150")
        )
        self.assertEqual(bootstrap.select_backend("cpu", "auto", rocm, dual_gpu), ("cpu", None))

    def test_unknown_amd_needs_an_architecture_instead_of_silent_cpu_fallback(self):
        amd = {"nvidia": False, "amd": True, "arch": None}
        with self.assertRaisesRegex(RuntimeError, "gfx1150"):
            bootstrap.select_backend("auto", "auto", {}, amd)
        self.assertEqual(bootstrap.select_backend("auto", "gfx1150", {}, amd), ("rocm", "gfx1150"))

    def test_explicit_backend_and_rocm_architecture_must_match_the_installed_runtime(self):
        self.assertTrue(bootstrap.compatible(installed_cuda(), "cuda", None))
        self.assertFalse(bootstrap.compatible(installed_cuda(), "cpu", None))
        self.assertFalse(bootstrap.compatible(installed_cuda(), "rocm", "gfx1150"))
        broken = installed_cuda()
        broken["available"] = False
        self.assertFalse(bootstrap.compatible(broken, "cuda", None))
        rocm = {**installed_cuda(), "backend": "rocm", "arch": "gfx1151"}
        self.assertFalse(bootstrap.compatible(rocm, "rocm", "gfx1150"))

    def test_wheel_commands_pin_the_backend_even_when_base_versions_match(self):
        python = Path("project with spaces") / "python.exe"
        cpu = bootstrap.torch_command(python, "cpu", None)
        self.assertIn("torch==2.11.0+cpu", cpu)
        self.assertIn("torchvision==0.26.0+cpu", cpu)
        amd = bootstrap.torch_command(python, "rocm", "gfx1150")
        self.assertIn("torch[device-gfx1150]==2.13.0+rocm10.0.0", amd)
        self.assertIn("torchvision[device-gfx1150]==0.28.0+rocm10.0.0", amd)
        self.assertEqual(amd[0], str(python))

    def test_app_dependencies_preserve_cuda_and_second_setup_skips_pip(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "env with spaces"
            python = bootstrap.python_in(venv)
            python.parent.mkdir(parents=True)
            python.touch()
            missing = installed_cuda(["cv2: missing"])
            ready = installed_cuda()
            with (
                patch.object(bootstrap, "run") as run,
                patch.object(bootstrap, "probe", return_value=ready),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                bootstrap.setup(venv, "cuda", None, missing, False)
                self.assertEqual(run.call_count, 1)
                command = run.call_args.args[0]
                self.assertIn("-c", command)
                self.assertNotIn("--index-url", command)
                self.assertEqual(
                    (venv / ".fastvideo-torch.txt").read_text(),
                    "torch==2.11.0+cu128\ntorchvision==0.26.0+cu128\n",
                )
                run.reset_mock()
                bootstrap.setup(venv, "cuda", None, ready, False)
                run.assert_not_called()

    def test_driver_failure_stops_before_app_installation_and_has_no_ready_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "env"
            python = bootstrap.python_in(venv)
            python.parent.mkdir(parents=True)
            python.touch()
            no_gpu = {**installed_cuda(), "available": False}
            with (
                patch.object(bootstrap, "run") as run,
                patch.object(bootstrap, "probe", return_value=no_gpu),
                self.assertRaisesRegex(RuntimeError, "pilotes"),
            ):
                bootstrap.setup(venv, "cuda", None, {}, False)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((venv / ".fastvideo-setup.json").exists())

    def test_dry_run_does_not_create_the_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "new env"
            output = io.StringIO()
            with patch.object(bootstrap, "run") as run, contextlib.redirect_stdout(output):
                bootstrap.setup(venv, "rocm", "gfx1150", {}, True)
            self.assertFalse(venv.exists())
            run.assert_not_called()
            self.assertIn("torch[device-gfx1150]", output.getvalue())

    def test_pip_failure_is_not_marked_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory) / "env"
            python = bootstrap.python_in(venv)
            python.parent.mkdir(parents=True)
            python.touch()
            with (
                patch.object(
                    bootstrap, "run", side_effect=subprocess.CalledProcessError(7, ["pip"])
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                bootstrap.setup(venv, "cuda", None, installed_cuda(["PIL: missing"]), False)
            self.assertFalse((venv / ".fastvideo-setup.json").exists())


if __name__ == "__main__":
    unittest.main()
