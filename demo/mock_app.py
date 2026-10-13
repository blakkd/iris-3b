"""Mock of the Iris-3B demo for local UI work: the exact UI of ``app.py`` with
no model weights, no GPU and no downloads.

    python demo/mock_app.py

"Generated" images are deterministic gradient compositions derived from the
prompt and seed, depth is a smoothed luminance field, and upscaling is a
bicubic 4x with a touch of sharpening. The precomputed example galleries
(``examples/depth``, ``examples/lowres``, ``examples/upscale``) are real.
Importing ``iris3b`` below only pulls in the library code (``fit_budget``,
``colorize``); nothing is loaded onto the GPU.
"""

import json
import os
import random
import time
import zlib
from pathlib import Path

import gradio as gr
import numpy as np
from matplotlib.colors import hsv_to_rgb
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from precompute_examples import DEPTH_EXAMPLES, UPSCALE_EXAMPLES
from ui import SIZES, build_demo, depth_view, meta, upscale_view

from iris3b.downstream.restoration import fit_budget

ROOT = Path(__file__).parent
PRESETS = json.loads((ROOT / "examples/presets.json").read_text())


def _stable(prompt: str, seed: int) -> int:
    return (int(seed) + zlib.crc32(prompt.encode("utf-8"))) % (2**31 - 1)


def fake_image(prompt: str, seed: int, height: int, width: int) -> Image.Image:
    """A deterministic abstract composition so prompt and seed changes are visible."""
    rng = np.random.default_rng(_stable(prompt, seed))
    hue = rng.uniform(0, 360)
    c1 = hsv_to_rgb([hue / 360, 0.55, 0.9])
    c2 = hsv_to_rgb([(hue + 120 + rng.uniform(0, 120)) / 360, 0.5, 0.7])
    yy, xx = np.mgrid[0:height, 0:width]
    t = 0.65 * xx / (width - 1) + 0.35 * yy / (height - 1)
    arr = (1 - t)[..., None] * c1 + t[..., None] * c2
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(9):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.uniform(0.05, 0.28) * min(width, height))
        value = int(rng.integers(140, 250)) if rng.random() > 0.4 else int(rng.integers(8, 90))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=value)
    layer = layer.filter(ImageFilter.GaussianBlur(max(8, min(width, height) // 12)))
    alpha = (np.asarray(layer, dtype=np.float32) / 255) * 0.3
    arr = arr * (1 - alpha[..., None]) + alpha[..., None] * arr.clip(0.3, 1.0)
    arr += rng.normal(0, 0.015, arr.shape)
    image = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            font = ImageFont.truetype(path, max(14, height // 30))
            break
    text = (prompt[:70] + "…") if len(prompt) > 70 else prompt
    y = height - font.size - 20
    draw.text((21, y + 2), text, font=font, fill=(0, 0, 0))  # shadow
    draw.text((20, y), text, font=font, fill=(255, 255, 255))
    return image


def mock_run(prompt, negative, size, steps, cfg_scale, seed, randomize):
    if randomize:
        seed = random.randint(0, 2**31 - 1)
    height, width = SIZES[size]
    time.sleep(min(1.5, 0.02 * int(steps)))  # visible pending state, like real sampling
    image = fake_image(prompt or "", seed, height, width)
    return image, seed, meta("Generated · mock", size, seed, steps, cfg_scale)


def fake_depth(image: Image.Image) -> np.ndarray:
    """A smooth relative log-depth field: brighter and lower = nearer."""
    width, height = image.size
    gray = Image.fromarray(np.asarray(image.convert("L"), dtype=np.uint8))
    field = np.asarray(gray.filter(ImageFilter.GaussianBlur(max(6, min(width, height) // 24))),
                        dtype=np.float32) / 255
    ramp = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]  # top of frame = farther
    depth = 0.55 * (1 - field) + 0.45 * ramp
    lo, hi = np.percentile(depth, [2, 98])
    return (2 * (depth - lo) / max(hi - lo, 1e-6) - 1).astype(np.float32)


def mock_depth(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    time.sleep(0.6)
    return depth_view(image, fake_depth(image), "Depth")


def mock_upscale(image):
    if image is None:
        raise gr.Error("Upload an image first.")
    small = fit_budget(image)
    time.sleep(0.9)
    result = small.resize((small.width * 4, small.height * 4), Image.Resampling.BICUBIC)
    result = ImageEnhance.Color(ImageEnhance.Sharpness(result).enhance(1.5)).enhance(1.08)
    note = f"{small.width}×{small.height} → {result.width}×{result.height}"
    return upscale_view(small, result, note)


demo = build_demo(
    run_fn=mock_run,
    depth_fn=mock_depth,
    upscale_fn=mock_upscale,
    presets=PRESETS,
    depth_examples=DEPTH_EXAMPLES,
    upscale_examples=UPSCALE_EXAMPLES,
    negative_prompt="",
    root=ROOT,
)

if __name__ == "__main__":
    demo.queue().launch(server_port=int(os.environ.get("PORT", 7860)))
