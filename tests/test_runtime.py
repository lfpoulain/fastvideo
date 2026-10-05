import re
import unittest
from types import SimpleNamespace

from vision import MODELS, choose_runtime


def fake_torch(available=True, hip=None, bf16=True):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            is_bf16_supported=lambda: bf16,
            get_device_name=lambda index: "Test GPU",
        ),
        version=SimpleNamespace(hip=hip),
        float16="fp16",
        bfloat16="bf16",
        float32="fp32",
    )


class RuntimeTests(unittest.TestCase):
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
