"""Configuration for data-free TDM distillation (arXiv:2503.06674).

The teacher, an exported directory or a training checkpoint, supplies the
model, flow and text-encoder sections. Distillation owns its optimizers, prompt
stream, step schedule and checkpoints here.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from iris3b.models.ac import POLICIES


@dataclass
class DMDConfig:
    """Distribution matching; ``sigmas`` is also the student's sampling schedule."""

    # Effective noise levels (1 = noise, 0 = data) at which the student's steps
    # start, strictly decreasing from 1. The clean endpoint 0 is implicit, so
    # each entry costs one student evaluation.
    sigmas: list[float] = field(default_factory=lambda: [1.0, 0.926, 0.833, 0.714, 0.556, 0.333])
    # guidance of the teacher target (empty negative prompt); the student samples without CFG
    cfg_scale: float = 3.5
    # floor of the matching noise level inside the last step, whose interval ends at 0
    sigma_min: float = 0.01

    def validate(self) -> None:
        if not self.sigmas or self.sigmas[0] != 1.0:
            raise ValueError("dmd.sigmas must start at pure noise (1.0)")
        if any(not math.isfinite(s) or not 0 < s <= 1 for s in self.sigmas):
            raise ValueError("dmd.sigmas must be finite and in (0, 1]; the clean endpoint 0 is implicit")
        if any(a <= b for a, b in zip(self.sigmas, self.sigmas[1:], strict=False)):
            raise ValueError("dmd.sigmas must be strictly decreasing")
        if not 0 < self.sigma_min < self.sigmas[-1]:
            raise ValueError("dmd.sigma_min must be in (0, last sigma)")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 1:
            raise ValueError("dmd.cfg_scale must be finite and >= 1")


@dataclass
class DistillConfig:
    teacher: str = ""  # exported directory (model.safetensors + config.yaml) or training .pth
    prompts_file: str = ""  # UTF-8 text, one prompt per line; blank lines are skipped
    work_dir: str = "output/distill"
    resume_from: str = ""  # a checkpoint of this run; continues it exactly
    # warm start: the student (init_weights ema | raw) and the fake network from
    # another distillation checkpoint, with fresh optimizers, counters and prompts
    init_from: str = ""
    init_weights: str = "ema"
    batch_size: int = 8  # per GPU
    height: int = 1024
    width: int = 1024
    # [height, width] buckets; every global batch draws one uniformly. Empty
    # trains at height x width only.
    resolutions: list[list[int]] = field(default_factory=list)
    max_steps: int = 1600  # student updates
    fake_updates: int = 1  # fake-network updates after every student update
    learning_rate: float = 1e-6
    fake_learning_rate: float = 5e-6
    betas: list[float] = field(default_factory=lambda: [0.0, 0.999])
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # 0 disables clipping
    ema_decay: float = 0.999
    activation_checkpointing: str = "full"  # none | full | selective_op | selective_layer
    # ddp keeps teacher, student, fake, EMA and both AdamW states in FP32 on every
    # GPU; fsdp2 shards student, fake and their optimizer states over all GPUs.
    # The frozen teacher and text encoder stay replicated in both.
    strategy: str = "ddp"  # ddp | fsdp2
    seed: int = 42
    save_every: int = 100
    log_every: int = 10
    dmd: DMDConfig = field(default_factory=DMDConfig)

    def validate(self) -> None:
        self.dmd.validate()
        for name in ("teacher", "prompts_file", "work_dir"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.init_weights not in ("ema", "raw"):
            raise ValueError("init_weights must be ema or raw")
        for name in ("batch_size", "height", "width", "max_steps", "fake_updates", "save_every", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(len(r) != 2 or min(r) <= 0 for r in self.resolutions):
            raise ValueError("resolutions must be [height, width] pairs of positive ints")
        for name in ("learning_rate", "fake_learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.betas) != 2 or any(not 0 <= b < 1 for b in self.betas):
            raise ValueError("betas must be two values in [0, 1)")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.activation_checkpointing not in POLICIES:
            raise ValueError(f"activation_checkpointing must be one of {', '.join(POLICIES)}")
        if self.strategy not in ("ddp", "fsdp2"):
            raise ValueError("strategy must be ddp or fsdp2")


def load_distill_config(path: str | Path, overrides: list[str] | None = None) -> DistillConfig:
    """Structured defaults <- YAML file <- dotlist overrides, validated."""
    conf = OmegaConf.merge(OmegaConf.structured(DistillConfig), OmegaConf.load(str(path)))
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.to_object(conf)
    cfg.validate()
    return cfg
