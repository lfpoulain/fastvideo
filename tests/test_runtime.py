import os
import re
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vision import MODEL_BY_KEY, MODELS, choose_runtime, runtime_diagnostics


def fake_torch(available=True, hip=None, bf16=True):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            is_bf16_supported=lambda: bf16,
            get_device_name=lambda index: "Test GPU",
            get_device_properties=lambda index: SimpleNamespace(gcnArchName="gfx1150"),
        ),
        __version__="test-version",
        version=SimpleNamespace(hip=hip),
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
    )


class RuntimeTests(unittest.TestCase):
    def test_rocm_diagnostics_report_permission_without_claiming_kernel_execution(self):
        for value, expected in [("1", "autorisés"), ("0", "désactivés")]:
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": value}),
            ):
                details = runtime_diagnostics(fake_torch(hip="7.2.1"), "cuda", MODELS[0])
                self.assertIn("HIP 7.2.1 · architecture gfx1150", details[0])
                self.assertIn(expected, details[1])

    def test_cpu_and_nvidia_diagnostics_do_not_claim_rocm_optimizations(self):
        for torch, device in [(fake_torch(), "cuda"), (fake_torch(hip="7.2.1"), "cpu")]:
            with self.subTest(device=device):
                details = runtime_diagnostics(torch, device, MODELS[0])
                self.assertEqual(details, ["Runtime · PyTorch test-version"])

    def test_lfm_fallback_diagnostic_uses_loaded_transformers_runtime(self):
        name = "transformers.models.lfm2.modeling_lfm2"
        for available, expected in [(False, "référence PyTorch"), (True, "disponibles")]:
            with (
                self.subTest(available=available),
                patch.dict(sys.modules, {name: SimpleNamespace(is_fast_path_available=available)}),
            ):
                details = runtime_diagnostics(fake_torch(), "cuda", MODEL_BY_KEY["lfm-3b"])
                self.assertIn(expected, details[-1])

    def test_lfm_python_wrappers_are_not_mistaken_for_compiled_kernels(self):
        name = "transformers.models.lfm2.modeling_lfm2"
        wrapped = SimpleNamespace(causal_conv1d_fn=lambda: None, causal_conv1d_update=lambda: None)
        for extension, expected in [(None, "référence PyTorch"), (wrapped, "disponibles")]:
            with (
                self.subTest(extension=extension),
                patch.dict(sys.modules, {name: wrapped, "causal_conv1d": extension}),
            ):
                details = runtime_diagnostics(fake_torch(), "cuda", MODEL_BY_KEY["lfm-3b"])
                self.assertIn(expected, details[-1])

    def test_rocm_uses_the_cuda_api_with_fp16(self):
        device, dtype, backend = choose_runtime(fake_torch(hip="10.0"))
        self.assertEqual(device, "cuda")
        self.assertEqual(dtype, "fp16")
        self.assertEqual(backend, "ROCm · Test GPU")

    def test_nvidia_uses_supported_precision(self):
        for supported, expected in [(True, "bf16"), (False, "fp16")]:
            with self.subTest(bf16=supported):
                device, dtype, backend = choose_runtime(fake_torch(bf16=supported))
                self.assertEqual((device, dtype), ("cuda", expected))
                self.assertEqual(backend, "CUDA · Test GPU")

    def test_cpu_is_available_as_fallback_or_explicit_choice(self):
        cases = [(fake_torch(available=False), "auto"), (fake_torch(hip="10.0"), "cpu")]
        for torch, requested in cases:
            with self.subTest(requested=requested):
                self.assertEqual(choose_runtime(torch, requested), ("cpu", "fp32", "CPU"))

    def test_explicit_gpu_choice_never_falls_back_silently(self):
        for requested, available, hip in [
            ("rocm", True, None),
            ("rocm", False, "10.0"),
            ("cuda", False, None),
            ("cuda", True, "10.0"),
        ]:
            with self.subTest(requested=requested, available=available, hip=hip):
                with self.assertRaises(RuntimeError):
                    choose_runtime(fake_torch(available=available, hip=hip), requested)

    def test_catalogue_has_unique_choices_and_fixed_revisions(self):
        expected = {
            "smol",
            "qwen-0.8b",
            "qwen-2b",
            "qwen-4b",
            "minicpm",
            "lfm-450m",
            "lfm-1.6b",
            "lfm-3b",
            "fastvlm",
            "fastvlm-1.5b",
            "fastvlm-7b",
            "moondream3",
        }
        self.assertEqual(expected, {spec.key for spec in MODELS})
        self.assertEqual(len({spec.key for spec in MODELS}), len(MODELS))
        self.assertEqual(len({spec.label for spec in MODELS}), len(MODELS))
        for spec in MODELS:
            with self.subTest(model=spec.key):
                self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{40}", spec.revision))
                self.assertIn(spec.max_frames, (1, 2, 3))
                if spec.adapter in ("fastvlm", "moondream"):
                    self.assertEqual(spec.max_frames, 1)
                if spec.tokenizer_repository:
                    self.assertIsNotNone(re.fullmatch(r"[0-9a-f]{40}", spec.tokenizer_revision))


if __name__ == "__main__":
    unittest.main()
