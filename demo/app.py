"""Iris-3B demo: text-to-image, monocular depth, image restoration and upscaling.

The UI itself lives in ``ui.py``; this module loads the three models and plugs
its inference callbacks into it. ``mock_app.py`` plugs in weight-free fakes of
the same callbacks for local UI work.
"""

import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from precompute_examples import DEPTH_EXAMPLES, UPSCALE_EXAMPLES  # outputs precomputed there
from ui import SIZES, build_demo, depth_view, meta, upscale_view

import iris3b.text  # noqa: F401  (registers encoders)
from iris3b.config import inference_config
from iris3b.downstream.depth import DepthPredictor
from iris3b.downstream.restoration import Restorer, fit_budget
from iris3b.models.dit import IrisDiT
from iris3b.registry import TEXT_ENCODERS
from iris3b.sampling import generate, load_for_inference

try:
    import spaces

    gpu = spaces.GPU
    ZERO_GPU = True
except ImportError:
    def gpu(duration):
        return lambda fn: fn
    ZERO_GPU = False


MODEL_REPO = os.environ.get("IRIS_MODEL_REPO", "speridlabs/iris-3b")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# All three models together are ~45 GB. On ZeroGPU (or with IRIS_OFFLOAD=1) they live on the CPU and
# only the one a request needs is moved to the GPU for that call.
HOME = torch.device("cpu") if ZERO_GPU or os.environ.get("IRIS_OFFLOAD") == "1" else DEVICE

source = MODEL_REPO if os.path.isdir(MODEL_REPO) else snapshot_download(
    MODEL_REPO, token=os.environ.get("HF_TOKEN"), ignore_patterns=["turbo/*"]  # Turbo has its own Space
)
raw, weights = load_for_inference(source)
cfg = inference_config(raw, [])
with torch.device("meta"):
    model = IrisDiT(cfg.model)
model.load_state_dict(weights, strict=True, assign=True)
model = model.eval().to(device=HOME, dtype=torch.float32)  # FP32 weights, BF16 autocast at inference
text_encoder = TEXT_ENCODERS.build(cfg.text_encoder.name, cfg.text_encoder, device=HOME)
depth_model = DepthPredictor(f"{source}/depth", device=HOME)
restorer = Restorer(f"{source}/upscaler", device=HOME)


@contextmanager
def on_gpu(*parts):
    for part in parts:
        part.to(DEVICE)
    try:
        yield
    finally:
        for part in parts:
            part.to(HOME)


@gpu(duration=120)
def infer(prompt, negative, size, steps, cfg_scale, seed, randomize):
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    height, width = SIZES[size]
    with on_gpu(model, text_encoder), torch.autocast(DEVICE.type, dtype=torch.bfloat16,
                                                     enabled=DEVICE.type == "cuda"):
        image = generate(
            model, text_encoder, [prompt], height=height, width=width, steps=int(steps), order=cfg.sample.order,
            cfg_scale=cfg_scale, cfg_interval=tuple(cfg.sample.cfg_interval), shift=cfg.flow.shift,
            negative_prompt=negative, generator=torch.Generator(device=DEVICE).manual_seed(int(seed)), device=DEVICE,
            num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction,
        )[0]
    pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels), seed


ROOT = Path(__file__).parent
# curated gallery images and the exact settings they were sampled with (100 steps, CFG 3.0)
PRESETS = json.loads((ROOT / "examples/presets.json").read_text())


def run(prompt, negative, size, steps, cfg_scale, seed, randomize):
    image, seed = infer(prompt, negative, size, steps, cfg_scale, seed, randomize)
    return image, seed, meta("Generated", size, seed, steps, cfg_scale)


@gpu(duration=60)
def run_depth(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    with on_gpu(depth_model):
        depth = depth_model(image)
    return depth_view(image, depth, "Depth")


@gpu(duration=120)
def run_upscale(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    small = fit_budget(image)
    with on_gpu(restorer):
        result = restorer(small)
    note = f"{small.width}×{small.height} → {result.width}×{result.height}"
    return upscale_view(small, result, note)


demo = build_demo(
    run_fn=run,
    depth_fn=run_depth,
    upscale_fn=run_upscale,
    presets=PRESETS,
    depth_examples=DEPTH_EXAMPLES,
    upscale_examples=UPSCALE_EXAMPLES,
    negative_prompt=cfg.sample.negative_prompt,
    root=ROOT,
)

if __name__ == "__main__":
    demo.queue().launch()
