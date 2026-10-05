import contextlib
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vision import CACHE_DIR, MODEL_BY_KEY, LocalVision


def engine_with_query(query):
    engine = LocalVision.__new__(LocalVision)
    engine.spec = MODEL_BY_KEY["moondream3"]
    engine.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    engine.model = SimpleNamespace(query=query)
    return engine


class MoondreamTests(unittest.TestCase):
    def test_query_uses_last_frame_without_reasoning_and_closes_stream(self):
        closed = []

        def chunks():
            try:
                yield " Une forme "
                yield "rouge. "
            finally:
                closed.append(True)

        query = Mock(return_value={"answer": chunks()})
        engine = engine_with_query(query)
        self.assertEqual(engine.describe(["old", "current"], "Décris", 17), "Une forme rouge.")
        query.assert_called_once_with(
            image="current",
            question="Décris",
            reasoning=False,
            stream=True,
            settings={"max_tokens": 17, "temperature": 0},
        )
        self.assertEqual(closed, [True])

    def test_cancellation_discards_partial_answer_and_releases_generator(self):
        cancel = threading.Event()
        closed = []

        def chunks():
            try:
                yield "partial "
                cancel.set()
                yield "discarded"
                raise AssertionError("The cancelled generator must not continue")
            finally:
                closed.append(True)

        query = Mock(return_value={"answer": chunks()})
        engine = engine_with_query(query)
        self.assertEqual(engine.describe(["image"], cancel=cancel), "")
        self.assertEqual(closed, [True])
        query.reset_mock()
        self.assertEqual(engine.describe(["image"], cancel=cancel), "")
        query.assert_not_called()

    def test_loader_pins_hidden_tokenizer_and_restores_module_even_on_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                engine = engine_with_query(Mock())
                engine.dtype, engine.device = "fp16", "cuda"
                engine.options = {
                    "cache_dir": CACHE_DIR,
                    "revision": engine.spec.revision,
                    "local_files_only": True,
                }
                tokenizer = object()
                original = object()
                vendor = SimpleNamespace(Tokenizer=original)
                model = Mock()
                encoder = model.model._vis_enc
                model.model.vision.pos_emb.dtype = "fp16"
                model.to.return_value = model
                model.eval.return_value = model

                def load(*args, **kwargs):
                    self.assertIs(
                        vendor.Tokenizer.from_pretrained("moondream/starmie-v1"), tokenizer
                    )
                    if fails:
                        raise RuntimeError("load failed")
                    return model

                model_class = SimpleNamespace(
                    __module__="vendor.hf_moondream", from_pretrained=Mock(side_effect=load)
                )
                download = Mock(return_value="pinned-tokenizer.json")
                from_file = Mock(return_value=tokenizer)
                dynamic = Mock(return_value=model_class)
                imports = {
                    "huggingface_hub": SimpleNamespace(hf_hub_download=download),
                    "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_file=from_file)),
                    "transformers.dynamic_module_utils": SimpleNamespace(
                        get_class_from_dynamic_module=dynamic
                    ),
                }
                with (
                    patch.dict("sys.modules", imports),
                    patch("vision.importlib.import_module", return_value=vendor),
                ):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "load failed"):
                            engine._load_moondream()
                    else:
                        engine._load_moondream()
                        self.assertFalse(engine.model.model.use_flex_decoding)
                        crops = Mock()
                        engine.model.model._vis_enc(crops)
                        crops.to.assert_called_once_with(dtype="fp16")
                        encoder.assert_called_once_with(crops.to.return_value)
                self.assertIs(vendor.Tokenizer, original)
                download.assert_called_once_with(
                    engine.spec.tokenizer_repository,
                    "tokenizer.json",
                    revision=engine.spec.tokenizer_revision,
                    cache_dir=CACHE_DIR,
                    local_files_only=True,
                )
                dynamic.assert_called_once_with(
                    "hf_moondream.HfMoondream", engine.spec.repository, **engine.options
                )
                model_class.from_pretrained.assert_called_once_with(
                    engine.spec.repository, **engine.options, use_safetensors=True, dtype="fp16"
                )


if __name__ == "__main__":
    unittest.main()
