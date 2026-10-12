"""The student's deterministic few-step rollout, as used in training."""

from collections.abc import Sequence

import torch

from iris3b.distill.objective import ModelFn, predict_x0


def backward_simulate(
    model_fn: ModelFn,
    noise: torch.Tensor,
    sigmas: Sequence[float],
    *,
    step_index: int,
    prediction: str = "v",
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the student from pure noise through step ``step_index``.

    Returns that step's clean prediction, its FP32 ``sigma[B]`` and its detached
    input state. Earlier steps run under no_grad and move each clean prediction
    to the next noise level along the flow, reusing the noise implied by the
    current state: the transition of ``iris3b.sampling.generate_few_step``. Only
    the selected step runs under the ambient autograd mode.
    """
    state = noise.detach().float()
    sigma = torch.full((state.shape[0],), sigmas[0], device=state.device, dtype=torch.float32)
    with torch.no_grad():
        for index in range(step_index):
            clean = predict_x0(model_fn, state, sigma, prediction, num_train_timesteps)
            current = sigma.view(-1, *([1] * (state.ndim - 1)))
            implied = (state - (1 - current) * clean) / current
            sigma = torch.full((state.shape[0],), sigmas[index + 1], device=state.device, dtype=torch.float32)
            following = sigma.view(-1, *([1] * (state.ndim - 1)))
            state = (1 - following) * clean + following * implied
    return predict_x0(model_fn, state, sigma, prediction, num_train_timesteps), sigma, state
