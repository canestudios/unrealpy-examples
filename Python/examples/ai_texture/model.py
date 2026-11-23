"""The text-to-image model of the example: Segmind-Vega (a 745M-parameter SDXL distillation) with the
Segmind-VegaRT LCM adapter (4 steps) and the fp16-safe SDXL VAE. All Apache-2.0 / MIT.

Runs only in the model process (worker.py), in its own environment (env/pyproject.toml). torch, diffusers and
huggingface_hub are imported inside the functions, so importing this module needs none of them.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from examples.ai_texture.core import Request, procedural_texture, texture_prompt

# The files the model needs and nothing more (no full-precision duplicates, no single-file checkpoints):
# 3,703 MB in all. Folder under the models directory -> (Hugging Face repository, file patterns).
MODEL_FILES: dict[str, tuple[str, list[str]]] = {
    "vega": (
        "segmind/Segmind-Vega",
        [
            "model_index.json",
            "scheduler/*",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "tokenizer/*",
            "tokenizer_2/*",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ],
    ),
    "vega_rt": ("segmind/Segmind-VegaRT", ["pytorch_lora_weights.safetensors"]),
    "vae_fp16_fix": ("madebyollin/sdxl-vae-fp16-fix", ["config.json", "diffusion_pytorch_model.safetensors"]),
}
# Where the weights live, chosen by the free VRAM at load (measured on an RTX 3070 Ti 8 GB, 1024 px):
# - "none": everything on the GPU, ~6.4 GB peak (3.1 GB of weights, 2.9 GB to decode), ~1.8 GB of RAM,
#   1.2-1.5 s per image;
# - "encoders": UNet and VAE stay on the GPU, the two text encoders (1.5 GB) wait in RAM and visit the GPU
#   for each prompt, leaving before the decode needs the room: ~5.3 GB peak, ~3.2 GB of RAM, 2.7-3 s;
# - "model": every component takes its turn on the GPU (model CPU offload): ~4.1 GB peak, ~4.8 GB of RAM,
#   3.9-5.3 s.
# Keep a margin: on Windows running out of VRAM does not fail, the driver spills into shared system memory
# and every image takes 30-120 s instead of one or two (seen at 6.9 GB free with everything on the GPU).
OFFLOAD_MODES = ("auto", "none", "encoders", "model")
NEEDS_FREE = {"none": 7.2 * 1024**3, "encoders": 6.0 * 1024**3}


def offload_plan(mode: str, free: int) -> str:
    """The OFFLOAD_MODES entry to load with: `mode` itself, or for "auto" the least offload that fits."""
    if mode not in OFFLOAD_MODES:
        raise ValueError(f"offload must be one of {', '.join(OFFLOAD_MODES)}, not {mode!r}")
    if mode != "auto":
        return mode
    return next((plan for plan, needed in NEEDS_FREE.items() if free >= needed), "model")


def fetch(models: Path, log: Callable[[str], None] = print) -> None:
    """Download MODEL_FILES into `models` (the only step that downloads anything)."""
    from huggingface_hub import snapshot_download  # ty: ignore[import-not-found]

    for folder, (repository, patterns) in MODEL_FILES.items():
        log(f"fetching {repository} -> {models / folder}")
        snapshot_download(repo_id=repository, allow_patterns=patterns, local_dir=models / folder)


def missing(models: Path) -> list[str]:
    """The model folders not downloaded yet."""
    return [folder for folder in MODEL_FILES if not (models / folder).is_dir()]


class VegaGenerator:
    """Loads the pipeline on the GPU once; generate() turns a Request into an H x W x 3 float image."""

    def __init__(self, models: Path, log: Callable[[str], None], offload: str = "auto") -> None:
        offload_plan(offload, 0)  # reject a bad mode before torch loads
        # Freed activations go back to the driver, not into a fragmented pool (must precede CUDA's start).
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from accelerate import cpu_offload_with_hook  # ty: ignore[import-not-found]
        from diffusers import (  # ty: ignore[import-not-found]
            AutoencoderKL,
            LCMScheduler,
            StableDiffusionXLPipeline,
            UNet2DConditionModel,
        )

        if not torch.cuda.is_available():
            # Never fall back to the CPU: fp32 weights in RAM are what the machine cannot afford.
            raise RuntimeError("no CUDA GPU visible to torch (is the CUDA build of torch installed?)")
        absent = missing(models)
        if absent:
            raise RuntimeError(f"model files missing in {models}: {', '.join(absent)} (run the fetch step)")
        self._torch = torch
        dtype = torch.float16
        free, total = torch.cuda.mem_get_info()
        self.plan = offload_plan(offload, free)
        self.offloaded = self.plan != "none"
        # What stays on the GPU goes straight from disk to VRAM (device_map) and never sits in RAM.
        gpu: dict[str, Any] = {"device_map": "cuda"}
        resident = gpu if self.plan != "model" else {}
        fp16: dict[str, Any] = {"dtype": dtype, "variant": "fp16", "use_safetensors": True}
        vae = AutoencoderKL.from_pretrained(models / "vae_fp16_fix", dtype=dtype, **resident)
        unet = UNet2DConditionModel.from_pretrained(models / "vega", subfolder="unet", **fp16, **resident)
        pipe: Any = StableDiffusionXLPipeline.from_pretrained(
            models / "vega", unet=unet, vae=vae, **fp16, **(gpu if self.plan == "none" else {})
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        pipe.load_lora_weights(str(models / "vega_rt"), weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora()
        pipe.unload_lora_weights()  # the fused weights stay; the adapter copies go
        self._encoders_hook: Any = None
        if self.plan == "encoders":
            # Each encoder moves to the GPU when called (the second sends the first back); generate() sends
            # the second back before the denoising and the decode.
            _, first = cpu_offload_with_hook(pipe.text_encoder, "cuda")
            _, self._encoders_hook = cpu_offload_with_hook(
                pipe.text_encoder_2, "cuda", prev_module_hook=first
            )
        elif self.plan == "model":
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._seamless: bool | None = None
        self.device = torch.cuda.get_device_name(0)
        placement = {
            "none": "fully on the GPU",
            "encoders": "text encoders offloaded to RAM",
            "model": "model CPU offload",
        }[self.plan]
        log(
            f"model on {self.device}: {free / 1024**3:.1f} of {total / 1024**3:.1f} GB VRAM free at load, "
            f"{placement}"
        )

    def _set_seamless(self, seamless: bool) -> None:
        """Circular padding in every convolution of the UNet and the VAE: the image wraps around its edges,
        so the texture tiles without seams."""
        if self._seamless == seamless:
            return
        conv = self._torch.nn.Conv2d
        for network in (self._pipe.unet, self._pipe.vae):
            for layer in network.modules():
                if isinstance(layer, conv):
                    layer.padding_mode = "circular" if seamless else "zeros"
        self._seamless = seamless

    def generate(self, request: Request) -> tuple[npt.NDArray[np.float32], int]:
        """The image (H x W x 3 floats in [0, 1]) and the seed used."""
        torch = self._torch
        seed = request.seed if request.seed >= 0 else secrets.randbelow(2**32)
        self._set_seamless(request.seamless)
        generator = torch.Generator("cuda").manual_seed(seed)
        try:
            with torch.inference_mode():
                prompt: dict[str, Any] = {"prompt": texture_prompt(request.prompt)}
                if self._encoders_hook is not None:
                    embeds, _, pooled, _ = self._pipe.encode_prompt(
                        prompt["prompt"], device="cuda", do_classifier_free_guidance=False
                    )
                    self._encoders_hook.offload()  # the text encoders leave before the decode needs the room
                    prompt = {"prompt_embeds": embeds, "pooled_prompt_embeds": pooled}
                result = self._pipe(
                    **prompt,
                    width=request.size,
                    height=request.size,
                    num_inference_steps=request.steps,
                    guidance_scale=0.0,  # the LCM adapter is distilled without classifier-free guidance
                    generator=generator,
                    output_type="np",
                )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"out of video memory at {request.size} px: try a smaller size") from exc
        finally:
            torch.cuda.empty_cache()  # the decode's ~3 GB of activations go back to the editor between images
        return np.asarray(result.images[0], dtype=np.float32), seed


class ProceduralGenerator:
    """The stand-in for VegaGenerator (`--fake`): numpy only, instant, always seamless."""

    device = "procedural (no model)"
    offloaded = False

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    def generate(self, request: Request) -> tuple[npt.NDArray[np.float32], int]:
        seed = request.seed if request.seed >= 0 else secrets.randbelow(2**32)
        if self._delay:
            time.sleep(self._delay)  # stands for GPU time, to see the editor stay responsive
        return procedural_texture(request.prompt, request.size, seed), seed
