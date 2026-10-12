"""Trajectory Distribution Matching losses (TDM, arXiv:2503.06674).

Iris uses ``x_sigma = (1 - sigma) * x0 + sigma * eps`` with velocity
``v = eps - x0``. Model closures take FP32 ``x`` and the model time
``sigma * num_train_timesteps`` and return the raw network output; prompt
conditioning and autocast belong to the closures.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from iris3b.distill.config import DMDConfig

ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class DMDLoss:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def _per_sample(value: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return value.view(-1, *([1] * (like.ndim - 1)))


def predict_x0(
    model_fn: ModelFn, x: torch.Tensor, sigma: torch.Tensor, prediction: str, num_train_timesteps: int
) -> torch.Tensor:
    """FP32 clean-image prediction at the per-example noise levels ``sigma[B]``."""
    x = x.float()
    raw = model_fn(x, sigma * num_train_timesteps).float()
    return raw if prediction == "x" else x - _per_sample(sigma, x) * raw


def tdm_noisy(
    clean: torch.Tensor,
    state: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: float,
    cfg: DMDConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Matching point of one deterministic student step.

    The step's next state x_next = (1 - s') x0 + s' eps_hat, with eps_hat the
    noise implied by the step input, is diffused forward to tau ~ U(s', s) by the
    rectified-flow transition x_tau = r x_next + sqrt(tau^2 - r^2 s'^2) eps,
    r = (1 - tau) / (1 - s'). Returns detached ``(x_tau, tau[B], n)`` with the
    effective noise n = (x_tau - (1 - tau) x0) / tau.
    """
    with torch.no_grad():
        clean = clean.detach().float()
        s = _per_sample(sigma, clean)
        implied = (state.float() - (1 - s) * clean) / s
        x_next = (1 - sigma_next) * clean + sigma_next * implied
        low = max(sigma_next, cfg.sigma_min)
        uniform = torch.rand(sigma.shape, device=clean.device, dtype=torch.float32, generator=generator)
        # tau keeps 2% of the step interval away from either end
        tau = low + (sigma - low) * (0.02 + 0.96 * uniform)
        t = _per_sample(tau, clean)
        r = (1 - t) / (1 - sigma_next)
        spread = (t.square() - (r * sigma_next).square()).clamp_min(0).sqrt()
        eps = torch.randn(clean.shape, device=clean.device, dtype=torch.float32, generator=generator)
        x_tau = r * x_next + spread * eps
        effective = (x_tau - (1 - t) * clean) / t
    return x_tau, tau, effective


def tdm_generator_loss(
    generated: torch.Tensor,
    state: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: float,
    real_cond_fn: ModelFn,
    real_uncond_fn: ModelFn,
    fake_cond_fn: ModelFn,
    cfg: DMDConfig,
    *,
    prediction: str = "v",
    num_train_timesteps: int = 1000,
    generator: torch.Generator | None = None,
) -> DMDLoss:
    """Stop-gradient proxy of the normalized TDM student update.

    The target moves the student's clean output by (x0_cfg - x0_fake) /
    mean|x0 - x0_cfg|, where x0_cfg is the guided teacher prediction at the
    matching point; the loss is 0.5 * mean((generated - target)^2).
    """
    generated = generated.float()
    with torch.no_grad():
        source = generated.detach()
        x_tau, tau, _ = tdm_noisy(source, state, sigma, sigma_next, cfg, generator)
        real = predict_x0(real_cond_fn, x_tau, tau, prediction, num_train_timesteps)
        fake = predict_x0(fake_cond_fn, x_tau, tau, prediction, num_train_timesteps)
        uncond = predict_x0(real_uncond_fn, x_tau, tau, prediction, num_train_timesteps)
        x0_cfg = uncond + cfg.cfg_scale * (real - uncond)
        dims = tuple(range(1, source.ndim))
        scale = (source - x0_cfg).abs().mean(dim=dims, keepdim=True).clamp_min(1e-6)
        target = source + (x0_cfg - fake) / scale
        metrics = {
            "dm_rms": ((real - fake) / scale).square().mean().sqrt(),
            "cfg_rms": ((x0_cfg - real) / scale).square().mean().sqrt(),
            "tau_mean": tau.mean(),
        }
    return DMDLoss(loss=0.5 * (generated - target).square().mean(), metrics=metrics)


def tdm_fake_loss(
    generated: torch.Tensor,
    state: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: float,
    fake_cond_fn: ModelFn,
    cfg: DMDConfig,
    *,
    prediction: str = "v",
    num_train_timesteps: int = 1000,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Clean-image denoising loss of the fake network at the matching point.

    The effective noise is not standard Gaussian, so every sample is
    importance-weighted by exp(-(mean(n^2) - 1) / 2), clamped to [0.2, 5].
    """
    target = generated.detach().float()
    x_tau, tau, effective = tdm_noisy(target, state, sigma, sigma_next, cfg, generator)
    dims = tuple(range(1, target.ndim))
    weight = torch.exp(-0.5 * (effective.square().mean(dim=dims) - 1.0)).clamp(0.2, 5.0)
    predicted = predict_x0(fake_cond_fn, x_tau, tau, prediction, num_train_timesteps)
    return (weight * (predicted - target).square().mean(dim=dims)).mean()
