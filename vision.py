"""Les huit modèles locaux et leur inférence, sans serveur ni API distante."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repository: str
    revision: str
    max_frames: int = 3
    adapter: str = "native"


MODELS = (
    ModelSpec(
        "smol",
        "SmolVLM2 · 500M",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    ),
    ModelSpec(
        "qwen-0.8b",
        "Qwen3.5 · 0,8B",
        "Qwen/Qwen3.5-0.8B",
        "2fc06364715b967f1860aea9cf38778875588b17",
    ),
    ModelSpec(
        "qwen-2b", "Qwen3.5 · 2B", "Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"
    ),
    ModelSpec(
        "minicpm",
        "MiniCPM-V · 4.6",
        "openbmb/MiniCPM-V-4.6",
        "36f34a661a4bd35d0dc2294cb044d2584646c7d3",
    ),
    ModelSpec(
        "lfm-450m",
        "LFM2.5-VL · 450M",
        "LiquidAI/LFM2.5-VL-450M",
        "fc6221ca597f3315e4f82fc2df606783267b34ba",
    ),
    ModelSpec(
        "lfm-1.6b",
        "LFM2.5-VL · 1,6B",
        "LiquidAI/LFM2.5-VL-1.6B",
        "919fde3d022e3f90a4716006f993938ee8c2eb97",
    ),
    ModelSpec(
        "lfm-3b",
        "LFM2.5-VL · 3B",
        "LiquidAI/LFM2.5-VL-3B",
        "35a118d938ce6d123ac2d371649f24a8efb69058",
    ),
    ModelSpec(
        "fastvlm",
        "FastVLM · 0,5B",
        "apple/FastVLM-0.5B",
        "16375720c2d673fa583e57e9876afde27549c7d0",
        max_frames=1,
        adapter="fastvlm",
    ),
)
MODEL_BY_KEY = {model.key: model for model in MODELS}
DEFAULT_MODEL = "smol"
CACHE_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_PROMPT = "Décris en français la scène et les objets visibles en deux phrases."


def make_messages(images, prompt):
    content = [{"type": "image", "image": image} for image in images]
    instruction = (
        "Observe uniquement les images fournies. "
        + (
            "Elles sont dans l'ordre chronologique ; compare les changements visibles. "
            if len(images) > 1
            else "Une seule image est fournie. "
        )
        + (prompt.strip() or DEFAULT_PROMPT)
    )
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def choose_runtime(torch, requested="auto"):
    """PyTorch utilise aussi le nom 'cuda' pour les GPU AMD avec ROCm."""
    available = torch.cuda.is_available()
    hip = bool(torch.version.hip)
    if requested in ("cuda", "rocm") and not available:
        raise RuntimeError("GPU indisponible : installe le PyTorch CUDA ou ROCm adapté à ton PC.")
    if requested == "rocm" and not hip:
        raise RuntimeError(
            "Ce PyTorch n'est pas une version ROCm. Utilise --device auto sur ce PC."
        )
    if requested == "cuda" and hip:
        raise RuntimeError("GPU AMD détecté : utilise --device auto ou --device rocm.")
    if requested == "cpu" or not available:
        return "cpu", torch.float32, "CPU"
    # FP16 sur ROCm : chemin commun aux petits VLM, sans kernels CUDA externes.
    dtype = torch.float16 if hip or not torch.cuda.is_bf16_supported() else torch.bfloat16
    backend = "ROCm" if hip else "CUDA"
    return "cuda", dtype, f"{backend} · {torch.cuda.get_device_name(0)}"


class LocalVision:
    def __init__(self, key=DEFAULT_MODEL, device="auto", offline=False):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.spec = MODEL_BY_KEY[key]
        self.torch = torch
        self.device, self.dtype, self.backend = choose_runtime(torch, device)
        self.options = dict(
            cache_dir=CACHE_DIR, revision=self.spec.revision, local_files_only=offline
        )
        self.model = self.processor = self.tokenizer = None
        if self.spec.adapter == "fastvlm":
            self._load_fastvlm()
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.spec.repository,
                **self.options,
                trust_remote_code=False,
            )
            self.model = (
                AutoModelForImageTextToText.from_pretrained(
                    self.spec.repository,
                    **self.options,
                    trust_remote_code=False,
                    use_safetensors=True,
                    dtype=self.dtype,
                    attn_implementation="sdpa",
                )
                .to(self.device)
                .eval()
            )

    def _load_fastvlm(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # FastVLM fournit son architecture dans le dépôt officiel Apple.
        # Le code téléchargé est fixé à une révision précise, inspectée pour cette app.
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.repository, **self.options)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.spec.repository,
                **self.options,
                trust_remote_code=True,
                use_safetensors=True,
                dtype=self.dtype,
                attn_implementation="sdpa",
            )
            .to(self.device)
            .eval()
        )
        self.processor = self.model.get_vision_tower().image_processor

    def describe(self, images, prompt=DEFAULT_PROMPT, max_tokens=100, cancel=None):
        from transformers import StoppingCriteria, StoppingCriteriaList

        if not images:
            raise ValueError("Aucune image à analyser.")
        if cancel is not None and cancel.is_set():
            return ""
        images = images[-self.spec.max_frames :]

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancel is not None and cancel.is_set()

        generation = dict(
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([Cancelled()]),
        )
        template_options = {}
        if self.spec.adapter == "native" and "enable_thinking" in str(self.processor.chat_template):
            template_options["enable_thinking"] = False
        processor_options = {}
        if self.spec.key == "minicpm":
            # Une vue par image : plus rapide et évite le merge de crops de tailles
            # différentes, non pris en charge dans Transformers 5.18.
            processor_options = {"images_kwargs": {"max_slice_nums": 1, "downsample_mode": "16x"}}
            generation["downsample_mode"] = "16x"
        with self.torch.inference_mode():
            if self.spec.adapter == "fastvlm":
                return self._describe_fastvlm(images[-1], prompt, generation)
            inputs = self.processor.apply_chat_template(
                make_messages(images, prompt),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=processor_options,
                **template_options,
            ).to(self.device, dtype=self.dtype)
            output = self.model.generate(**inputs, **generation)
            return self.processor.decode(
                output[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

    def _describe_fastvlm(self, image, prompt, generation):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<image>\n" + (prompt.strip() or DEFAULT_PROMPT)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        before, after = rendered.split("<image>", 1)
        ids = (
            self.tokenizer.encode(before, add_special_tokens=False)
            + [-200]
            + self.tokenizer.encode(after, add_special_tokens=False)
        )
        input_ids = self.torch.tensor([ids], device=self.device)
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].to(
            self.device,
            dtype=self.dtype,
        )
        output = self.model.generate(
            inputs=input_ids,
            attention_mask=self.torch.ones_like(input_ids),
            images=pixels,
            **generation,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def close(self):
        """Libérer le précédent modèle avant de charger le suivant."""
        self.model = self.processor = self.tokenizer = None
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


def describe_timed(engine, images, prompt, max_tokens, cancel=None):
    engine.synchronize()
    start = time.perf_counter()
    description = engine.describe(images, prompt, max_tokens, cancel)
    engine.synchronize()
    return description, time.perf_counter() - start
