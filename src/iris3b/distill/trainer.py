"""Prompt-only TDM distillation of Iris-3B, replicated (DDP) or sharded (FSDP2).

Three networks share the teacher's architecture: the frozen teacher, the
student and a "fake" score network trained online on the student's outputs.
Every student update is followed by ``fake_updates`` fake updates. Checkpoints
hold full (unsharded) tensors and are written only between complete cycles, so
a resume continues the run exactly, under either strategy.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from torch import nn

from iris3b.distill.config import DistillConfig
from iris3b.distill.objective import tdm_fake_loss, tdm_generator_loss
from iris3b.distill.sampler import backward_simulate
from iris3b.train.ema import EMA, ShardEMA, _local

_FORMAT_VERSION = 1
# Settings a resume may change: locations and the warm start (the checkpoint
# already holds its effect), reporting cadence, the endpoint, the strategy
# (checkpoints hold full tensors either way) and the step sizes, whose live
# values are written into the restored optimizers.
_RESUME_MUTABLE = {
    "teacher",
    "prompts_file",
    "work_dir",
    "resume_from",
    "init_from",
    "init_weights",
    "strategy",
    "max_steps",
    "save_every",
    "log_every",
    "learning_rate",
    "fake_learning_rate",
    "max_grad_norm",
}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_content_hash(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over the names, dtypes, shapes and bytes of a state dict."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(memoryview(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


@dataclass
class PromptStream:
    """Cyclic global prompt stream; each rank takes a contiguous slice of every batch."""

    prompts: tuple[str, ...]
    sha256: str
    cursor: int = 0  # prompts consumed globally, by student and fake updates

    @classmethod
    def from_file(cls, path: str | Path) -> PromptStream:
        content = Path(path).read_bytes()
        prompts = tuple(line for line in content.decode("utf-8").splitlines() if line.strip())
        if not prompts:
            raise ValueError("prompts_file must contain at least one nonblank UTF-8 prompt")
        return cls(prompts, hashlib.sha256(content).hexdigest())

    def take(self, batch_size: int, rank: int, world_size: int) -> list[str]:
        start = self.cursor + rank * batch_size
        batch = [self.prompts[(start + offset) % len(self.prompts)] for offset in range(batch_size)]
        self.cursor += batch_size * world_size
        return batch

    def state_dict(self) -> dict:
        return {"sha256": self.sha256, "count": len(self.prompts), "cursor": self.cursor}


@dataclass(frozen=True)
class Conditioning:
    embeddings: torch.Tensor
    mask: torch.Tensor
    null_embeddings: torch.Tensor
    null_mask: torch.Tensor


class _TensorOutput(nn.Module):
    """Return the prediction tensor itself, so DDP sees a tensor output."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x, t, y, *, y_mask):
        return self.core(x, t, y, y_mask=y_mask).x


@contextmanager
def _evaluating(module: nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


def _collective_error(error: Exception | None, world_size: int) -> None:
    """Raise a local failure on every rank before peers enter another collective."""
    if world_size == 1:
        if error is not None:
            raise error
        return
    errors = [None] * world_size
    torch.distributed.all_gather_object(errors, None if error is None else f"{type(error).__name__}: {error}")
    if any(item is not None for item in errors):
        detail = "; ".join(f"rank {rank}: {item}" for rank, item in enumerate(errors) if item is not None)
        raise RuntimeError(f"distillation failed: {detail}")


def shard_trainable(model: nn.Module, mesh, wrap_names: set[str]) -> None:
    """FSDP2-shard the student or fake in place, per block and then at the root.

    The default mixed-precision policy keeps FP32 parameters and reductions and
    leaves every input in the dtype it arrives in, so under autocast the sharded
    module computes exactly what the unsharded teacher does. Casting block inputs
    to the parameter dtype would bias the teacher-minus-fake term from the first
    update.
    """
    from torch.distributed.fsdp import fully_shard

    for block in [module for module in model.modules() if type(module).__name__ in wrap_names]:
        fully_shard(block, mesh=mesh)
    # the root group (embedders and head) stays gathered after forward: the
    # output is a dataclass FSDP2 cannot hook to re-gather the group for backward
    fully_shard(model, mesh=mesh, reshard_after_forward=False)


def _is_sharded(tensor: torch.Tensor) -> bool:
    from torch.distributed.tensor import DTensor

    return isinstance(tensor, DTensor)


def _shard_like(full: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """``full`` laid out like ``live``: the same DTensor placement, or as is."""
    full = full.to(device=live.device, dtype=live.dtype)
    if not _is_sharded(live):
        return full
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(full, live.device_mesh, live.placements)


@torch.no_grad()
def _load_full_state(model: nn.Module, full: Mapping[str, torch.Tensor]) -> None:
    """Copy a full state dict into a sharded module, shard by shard."""
    live = model.state_dict()
    if live.keys() != full.keys():
        raise ValueError("checkpoint state dict keys differ from the model")
    for name, tensor in live.items():
        _local(tensor).copy_(_local(_shard_like(full[name], tensor)))


def _load_full_optimizer(optimizer: torch.optim.Optimizer, saved: dict) -> None:
    """Shard a full AdamW state dict onto the live DTensor parameters, by position."""
    for saved_group, live_group in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        for index, parameter in zip(saved_group["params"], live_group["params"], strict=True):
            if index not in saved["state"]:
                continue
            moments = saved["state"][index]
            step = moments["step"]
            optimizer.state[parameter] = {
                "step": step.clone() if isinstance(step, torch.Tensor) else step,
                "exp_avg": _shard_like(moments["exp_avg"], parameter),
                "exp_avg_sq": _shard_like(moments["exp_avg_sq"], parameter),
            }


def _training_contract(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in _RESUME_MUTABLE}


def _capture_rng(device: torch.device, generator: torch.Generator) -> dict:
    import numpy as np

    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
        "generator": generator.get_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(state: dict, device: torch.device, generator: torch.Generator) -> None:
    import numpy as np

    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)
    generator.set_state(state["generator"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


class DistillationState:
    """The student/fake updates, EMA and the full-resume checkpoint protocol.

    Student, fake and teacher are independent FP32 modules on ``device``. Student
    and fake are either both ``fully_shard``-ed (FSDP2, DTensor parameters, no
    wrapper) or optionally wrapped as ``DDP(_TensorOutput(core))``; no-grad
    calls always use the bare modules.
    """

    def __init__(
        self,
        cfg: DistillConfig,
        *,
        student: nn.Module,
        fake: nn.Module,
        teacher: nn.Module,
        text_encoder,
        prompts: PromptStream,
        base_config: dict,
        teacher_identity: dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
        student_wrapper: nn.Module | None = None,
        fake_wrapper: nn.Module | None = None,
    ):
        self.cfg = cfg
        self.device = device
        self.rank, self.world_size = rank, world_size
        self.student, self.fake, self.teacher = student, fake, teacher
        self.sharded = any(_is_sharded(p) for p in student.parameters())
        self.student_wrapper = student_wrapper or _TensorOutput(student)
        self.fake_wrapper = fake_wrapper or _TensorOutput(fake)
        self.text_encoder, self.prompts = text_encoder, prompts
        null = text_encoder.null("")
        self.null = (null.embeddings.to(device=device, dtype=torch.float32), null.mask.to(device=device))
        self.base_config = base_config
        self.teacher_identity = teacher_identity
        self.prediction = base_config["flow"]["prediction"]
        self.num_train_timesteps = base_config["flow"]["num_train_timesteps"]
        self.in_channels = base_config["model"]["in_channels"]
        self.generator = torch.Generator(device=device).manual_seed(cfg.seed + rank)
        optimizer_kwargs = {"betas": tuple(cfg.betas), "weight_decay": cfg.weight_decay}
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.learning_rate, **optimizer_kwargs)
        self.fake_optimizer = torch.optim.AdamW(
            fake.parameters(), lr=cfg.fake_learning_rate, **optimizer_kwargs
        )
        # sharded: the shadow is per shard, and ShardEMA.update must follow optimizer.step()
        self.ema = ShardEMA(student, cfg.ema_decay) if self.sharded else EMA(student, cfg.ema_decay)
        self.generator_updates = 0
        self.fake_updates = 0

    def _model_fn(self, core: nn.Module, conditioning: Conditioning, *, uncond=False, wrapper=None):
        embeddings = conditioning.null_embeddings if uncond else conditioning.embeddings
        mask = conditioning.null_mask if uncond else conditioning.mask

        def call(x, t_model):
            differentiable = wrapper is not None and torch.is_grad_enabled()
            mode = nullcontext() if differentiable else _evaluating(core)
            with mode, torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                if differentiable:
                    output = wrapper(x, t_model, embeddings, y_mask=mask)
                else:
                    output = core(x, t_model, embeddings, y_mask=mask).x
                    if self.sharded and core is not self.teacher:
                        # the FSDP2 root group stays gathered after forward and
                        # only a backward reshards it; without one, the optimizer
                        # step, EMA and save would see stale unsharded parameters
                        core.reshard()
            return output.float()

        return call

    def _conditioning(self) -> Conditioning:
        prompts = self.prompts.take(self.cfg.batch_size, self.rank, self.world_size)
        encoding = self.text_encoder.encode(prompts)
        null_embeddings, null_mask = self.null
        return Conditioning(
            encoding.embeddings.to(device=self.device, dtype=torch.float32),
            encoding.mask.to(device=self.device),
            null_embeddings.expand(len(prompts), *([-1] * (null_embeddings.ndim - 1))),
            null_mask.expand(len(prompts), -1),
        )

    def _generate(self, conditioning: Conditioning, *, differentiable: bool):
        """Student rollout to a uniformly drawn step, at a uniformly drawn bucket.

        Returns the step's clean prediction, its sigma[B], its input state and the
        next step's sigma (0 after the last step).
        """
        sigmas = self.cfg.dmd.sigmas
        buckets = self.cfg.resolutions or [[self.cfg.height, self.cfg.width]]
        choice = torch.stack(
            [
                torch.randint(len(sigmas), (), device=self.device, generator=self.generator),
                torch.randint(len(buckets), (), device=self.device, generator=self.generator),
            ]
        )
        if self.world_size > 1:
            # every rank advances its own generator; rank 0 picks the step (the
            # depth of the backward graph) and bucket, so all ranks run one graph
            torch.distributed.broadcast(choice, src=0)
        step, bucket = (int(value) for value in choice.tolist())
        height, width = buckets[bucket]
        noise = torch.randn(
            self.cfg.batch_size,
            self.in_channels,
            height,
            width,
            device=self.device,
            dtype=torch.float32,
            generator=self.generator,
        )
        model_fn = self._model_fn(
            self.student, conditioning, wrapper=self.student_wrapper if differentiable else None
        )
        with torch.enable_grad() if differentiable else torch.no_grad():
            generated, sigma, state = backward_simulate(
                model_fn,
                noise,
                sigmas,
                step_index=step,
                prediction=self.prediction,
                num_train_timesteps=self.num_train_timesteps,
            )
        return generated, sigma, state, sigmas[step + 1] if step + 1 < len(sigmas) else 0.0

    def _require_finite(self, value: torch.Tensor, label: str) -> None:
        finite = torch.isfinite(value.detach()).all().to(dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"nonfinite {label} on at least one rank; no optimizer step was taken")

    def _optimize(
        self, loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> torch.Tensor:
        self._require_finite(loss, "loss")
        loss.backward()
        gradients = [_local(p.grad) for p in model.parameters() if p.grad is not None]
        if not gradients:
            norm = torch.full((), float("nan"), device=self.device)
        elif self.sharded:
            # every gradient is Shard(0) over all ranks, so the squared norms of
            # the local shards sum to the full squared norm
            squares = torch.stack([g.float().square().sum() for g in gradients]).sum()
            if self.world_size > 1:
                torch.distributed.all_reduce(squares)
            norm = squares.sqrt()
        else:
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
        self._require_finite(norm, "gradient norm (or missing gradients)")
        if self.cfg.max_grad_norm > 0:
            torch._foreach_mul_(gradients, (self.cfg.max_grad_norm / (norm + 1e-6)).clamp(max=1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return norm.detach()

    def student_step(self) -> dict[str, torch.Tensor]:
        self.student.train()
        self.fake.eval()
        self.optimizer.zero_grad(set_to_none=True)
        self.fake_optimizer.zero_grad(set_to_none=True)
        conditioning = self._conditioning()
        generated, sigma, state, sigma_next = self._generate(conditioning, differentiable=True)
        result = tdm_generator_loss(
            generated,
            state,
            sigma,
            sigma_next,
            self._model_fn(self.teacher, conditioning),
            self._model_fn(self.teacher, conditioning, uncond=True),
            self._model_fn(self.fake, conditioning),
            self.cfg.dmd,
            prediction=self.prediction,
            num_train_timesteps=self.num_train_timesteps,
            generator=self.generator,
        )
        norm = self._optimize(result.loss, self.student, self.optimizer)
        self.ema.update() if self.sharded else self.ema.update(self.student)
        self.generator_updates += 1
        return {"generator_loss": result.loss.detach(), "generator_grad_norm": norm, **result.metrics}

    def fake_step(self) -> dict[str, torch.Tensor]:
        self.fake.train()
        self.fake_optimizer.zero_grad(set_to_none=True)
        # a fresh prompt batch, step and noise for every fake update
        conditioning = self._conditioning()
        generated, sigma, state, sigma_next = self._generate(conditioning, differentiable=False)
        loss = tdm_fake_loss(
            generated,
            state,
            sigma,
            sigma_next,
            self._model_fn(self.fake, conditioning, wrapper=self.fake_wrapper),
            self.cfg.dmd,
            prediction=self.prediction,
            num_train_timesteps=self.num_train_timesteps,
            generator=self.generator,
        )
        norm = self._optimize(loss, self.fake, self.fake_optimizer)
        self.fake_updates += 1
        return {"fake_loss": loss.detach(), "fake_grad_norm": norm}

    def update(self) -> dict[str, torch.Tensor]:
        self._require_boundary()
        metrics = self.student_step()
        fake_metrics = [self.fake_step() for _ in range(self.cfg.fake_updates)]
        metrics.update(
            {key: torch.stack([row[key] for row in fake_metrics]).mean() for key in fake_metrics[0]}
        )
        return {key: value.detach() for key, value in metrics.items()}

    def _require_boundary(self) -> None:
        if self.fake_updates != self.generator_updates * self.cfg.fake_updates:
            raise ValueError("checkpoint/update requires a complete student-plus-fake cycle")
        expected = (self.generator_updates + self.fake_updates) * self.cfg.batch_size * self.world_size
        if self.prompts.cursor != expected:
            raise ValueError("prompt cursor does not match the completed optimizer updates")

    def save(self, path: str | Path) -> Path:
        from iris3b.train.ckpt import gather_full_state, save_checkpoint

        self._require_boundary()
        local = {
            "rank": self.rank,
            "rng": _capture_rng(self.device, self.generator),
            "counters": (self.generator_updates, self.fake_updates, self.prompts.cursor),
        }
        ranks = [local]
        if self.world_size > 1:
            ranks = [None] * self.world_size
            torch.distributed.all_gather_object(ranks, local)
        if any(item["counters"] != local["counters"] for item in ranks):
            raise ValueError("ranks disagree about update counters or prompt cursor")
        error = None
        if self.sharded:
            # collective: every rank gathers, only rank 0 keeps the full copies
            keep = self.rank == 0
            student = gather_full_state(self.student.state_dict(), keep)
            with self.ema.swapped():
                ema = gather_full_state(self.student.state_dict(), keep)
            fake = gather_full_state(self.fake.state_dict(), keep)
            optimizer = gather_full_state(self.optimizer.state_dict(), keep)
            fake_optimizer = gather_full_state(self.fake_optimizer.state_dict(), keep)
        if self.rank == 0:
            try:
                if not self.sharded:
                    student, ema, fake = (
                        self.student.state_dict(),
                        self.ema.state_dict(),
                        self.fake.state_dict(),
                    )
                    optimizer, fake_optimizer = self.optimizer.state_dict(), self.fake_optimizer.state_dict()
                save_checkpoint(
                    path,
                    student,
                    optimizer=optimizer,
                    ema=ema,
                    epoch=0,
                    step=self.generator_updates,
                    config_dict=self.base_config,
                    extra_state={
                        "distill_format_version": _FORMAT_VERSION,
                        "checkpoint_id": uuid.uuid4().hex,
                        "distill_config": asdict(self.cfg),
                        "teacher_identity": self.teacher_identity,
                        "state_dict_fake": fake,
                        "fake_optimizer": fake_optimizer,
                        "generator_updates": self.generator_updates,
                        "fake_updates": self.fake_updates,
                        "world_size": self.world_size,
                        "prompt_state": self.prompts.state_dict(),
                        "rank_states": ranks,
                    },
                )
            except Exception as exc:
                error = exc
        _collective_error(error, self.world_size)
        return Path(path)

    def _check_resume(self, payload: dict) -> None:
        if payload.get("distill_format_version") != _FORMAT_VERSION:
            raise ValueError("not a checkpoint written by this distillation trainer")
        if payload["teacher_identity"] != self.teacher_identity or payload["config"] != self.base_config:
            raise ValueError("resume teacher weights or config differ")
        if _training_contract(payload["distill_config"]) != _training_contract(asdict(self.cfg)):
            raise ValueError(
                "resume settings differ; only " + ", ".join(sorted(_RESUME_MUTABLE)) + " may change"
            )
        if payload["world_size"] != self.world_size:
            raise ValueError("an exact resume requires the same world size")
        if payload["prompt_state"]["sha256"] != self.prompts.sha256:
            raise ValueError("resume prompt file content differs")
        if payload["generator_updates"] > self.cfg.max_steps:
            raise ValueError("resume checkpoint is past max_steps")

    def load(self, path: str | Path) -> None:
        error, payload = None, None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            self._check_resume(payload)
        except Exception as exc:
            error = exc
        _collective_error(error, self.world_size)
        if self.world_size > 1:
            identities = [None] * self.world_size
            torch.distributed.all_gather_object(identities, payload["checkpoint_id"])
            if len(set(identities)) != 1:
                raise ValueError("ranks loaded different resume checkpoints")
        try:
            if self.sharded:
                # seed the per-shard EMA by loading the averaged weights first
                _load_full_state(self.student, payload["state_dict_ema"])
                self.ema = ShardEMA(self.student, self.cfg.ema_decay)
                _load_full_state(self.student, payload["state_dict"])
                _load_full_state(self.fake, payload["state_dict_fake"])
                _load_full_optimizer(self.optimizer, payload["optimizer"])
                _load_full_optimizer(self.fake_optimizer, payload["fake_optimizer"])
            else:
                self.student.load_state_dict(payload["state_dict"], strict=True)
                self.fake.load_state_dict(payload["state_dict_fake"], strict=True)
                self.ema.load_state_dict(payload["state_dict_ema"])
                self.optimizer.load_state_dict(payload["optimizer"])
                self.fake_optimizer.load_state_dict(payload["fake_optimizer"])
            self.generator_updates = payload["generator_updates"]
            self.fake_updates = payload["fake_updates"]
            for optimizer, lr in (
                (self.optimizer, self.cfg.learning_rate),
                (self.fake_optimizer, self.cfg.fake_learning_rate),
            ):
                for group in optimizer.param_groups:
                    group["lr"] = lr
            self.prompts.cursor = payload["prompt_state"]["cursor"]
            # last, so that loading and collectives cannot consume replayed randomness
            _restore_rng(payload["rank_states"][self.rank]["rng"], self.device, self.generator)
        except Exception as exc:
            error = exc
        _collective_error(error, self.world_size)


class DistillationTrainer:
    def __init__(self, cfg: DistillConfig):
        cfg.validate()
        self.cfg = cfg

    def run(self) -> None:
        import numpy as np
        from torch.nn.parallel import DistributedDataParallel

        import iris3b.text  # noqa: F401  (registers text encoders)
        from iris3b.config import INFERENCE_SECTIONS, inference_config
        from iris3b.models import IrisDiT
        from iris3b.registry import TEXT_ENCODERS
        from iris3b.sampling import load_for_inference

        cfg = self.cfg
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("distillation requires CUDA with bf16 support")
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        created_group = False
        try:
            # fsdp2 needs a process group even on a single GPU
            if (world_size > 1 or cfg.strategy == "fsdp2") and not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=30))
                created_group = True
            random.seed(cfg.seed + rank)
            np.random.seed((cfg.seed + rank) % 2**32)
            torch.manual_seed(cfg.seed + rank)
            torch.cuda.manual_seed(cfg.seed + rank)
            torch.backends.cudnn.benchmark = False
            error = None
            try:
                raw, weights = load_for_inference(cfg.teacher)
                base = inference_config(raw)
                patch = base.model.patch_size
                if any(h % patch or w % patch for h, w in cfg.resolutions or [[cfg.height, cfg.width]]):
                    raise ValueError(
                        f"every training resolution must be divisible by model.patch_size={patch}"
                    )
                base_config = {name: asdict(getattr(base, name)) for name in INFERENCE_SECTIONS}
                identity = {
                    "weights_sha256": tensor_content_hash(weights),
                    "config_sha256": _json_hash(base_config),
                }
                with torch.device("meta"):
                    student = IrisDiT(base.model)
                student.load_state_dict(weights, strict=True, assign=True)
                del weights
                # FP32 weights under BF16 autocast for all three networks: a bf16
                # teacher would shift its x0 against the FP32 fake and bias their
                # difference
                student.to(device=device, dtype=torch.float32)
                teacher = deepcopy(student).requires_grad_(False).eval()
                fake = deepcopy(student)
                if cfg.init_from and not cfg.resume_from:
                    payload = torch.load(cfg.init_from, map_location="cpu", weights_only=False, mmap=True)
                    key = "state_dict_ema" if cfg.init_weights == "ema" else "state_dict"
                    student.load_state_dict(payload[key], strict=True)
                    fake.load_state_dict(payload["state_dict_fake"], strict=True)
                    del payload
                student.activation_checkpointing = cfg.activation_checkpointing
                fake.activation_checkpointing = cfg.activation_checkpointing
                prompts = PromptStream.from_file(cfg.prompts_file)
                text_encoder = TEXT_ENCODERS.build(base.text_encoder.name, base.text_encoder, device)
                if rank == 0:
                    text_encoder.null("")  # fills the on-disk null cache before the other ranks read it
            except Exception as exc:
                error = exc
            _collective_error(error, world_size)
            if world_size > 1:
                identities = [None] * world_size
                signature = (identity, prompts.sha256, asdict(cfg))
                torch.distributed.all_gather_object(identities, signature)
                if any(item != signature for item in identities):
                    raise ValueError("ranks disagree about teacher, prompts or distillation config")
            wrappers = [None, None]
            if cfg.strategy == "fsdp2":
                from torch.distributed.device_mesh import init_device_mesh

                from iris3b.train.parallel import wrap_class_names

                mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
                for model in (student, fake):
                    shard_trainable(model, mesh, wrap_class_names(base.model))
            elif world_size > 1:
                # unused-parameter discovery covers architectures whose final block
                # computes a text output that nothing consumes
                wrappers = [
                    DistributedDataParallel(
                        _TensorOutput(model),
                        device_ids=[local_rank],
                        output_device=local_rank,
                        find_unused_parameters=True,
                    )
                    for model in (student, fake)
                ]
            state = DistillationState(
                cfg,
                student=student,
                fake=fake,
                teacher=teacher,
                text_encoder=text_encoder,
                prompts=prompts,
                base_config=base_config,
                teacher_identity=identity,
                device=device,
                rank=rank,
                world_size=world_size,
                student_wrapper=wrappers[0],
                fake_wrapper=wrappers[1],
            )
            if cfg.resume_from:
                state.load(cfg.resume_from)
            if rank == 0:
                start = {
                    "event": "distill_start",
                    "world_size": world_size,
                    "strategy": cfg.strategy,
                    "global_batch": cfg.batch_size * world_size,
                    "generator_updates": state.generator_updates,
                    "fake_updates_per_generator": cfg.fake_updates,
                    "teacher_identity": identity,
                    "sigmas": cfg.dmd.sigmas,
                }
                print(json.dumps(start, sort_keys=True), flush=True)
            checkpoints = Path(cfg.work_dir) / "checkpoints"
            last_saved = None
            while state.generator_updates < cfg.max_steps:
                metrics = state.update()
                step = state.generator_updates
                if step % cfg.log_every == 0 or step == 1 or step == cfg.max_steps:
                    names = sorted(metrics)
                    values = torch.stack([metrics[name].float() for name in names])
                    if world_size > 1:
                        torch.distributed.all_reduce(values)
                        values /= world_size
                    if rank == 0:
                        row = dict(zip(names, values.cpu().tolist(), strict=True))
                        row.update(
                            step=step, fake_updates=state.fake_updates, prompt_cursor=state.prompts.cursor
                        )
                        print(json.dumps(row, sort_keys=True), flush=True)
                if step % cfg.save_every == 0 or step == cfg.max_steps:
                    state.save(checkpoints / f"step_{step:08d}.pth")
                    last_saved = step
            if last_saved != state.generator_updates:
                state.save(checkpoints / f"step_{state.generator_updates:08d}.pth")
        finally:
            if created_group:
                torch.distributed.destroy_process_group()
