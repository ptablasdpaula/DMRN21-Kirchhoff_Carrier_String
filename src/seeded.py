"""Experimental coordinate-seeded forward sensitivities for a scalar loss."""
import torch
from torch.autograd.function import once_differentiable
from .accelerated import evaluate
from .kc_string_forward import nominal_parameters


class _SeededAudio(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, theta, seeds, indices, model, steps):
        audio, tangents = evaluate(theta, model.intervals, model.sample_rate, steps,
            model.output_stride, model.excitation, model.backend, True,
            model.chunk_steps, model.threads, seeds=seeds, order=model.order)
        ctx.save_for_backward(tangents, indices)
        ctx.count = values.numel()
        return audio

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        tangents, indices = ctx.saved_tensors
        grad = grad_output.new_zeros(ctx.count)
        grad = grad.scatter(0, indices, grad_output @ tangents)
        return grad, None, None, None, None, None


def seeded_audio(values, box, model, duration, mask):
    """Output with derivatives in selected *physical coordinate* directions.

    The seed d(material multiplier)/d(coordinate) is evaluated at the current
    parameters. The custom backward supplies those selected coordinate gradients;
    the caller's bounded-coordinate transform remains in the ordinary graph.
    """
    if model.backend not in ('modal', 'spatial'):
        raise ValueError('Seeded sensitivity requires the modal or spatial backend.')
    if len(mask)!=values.numel() or not any(mask):
        raise ValueError('Select at least one coordinate for seeded sensitivity.')
    indices = torch.tensor([i for i, enabled in enumerate(mask) if enabled],
                           device=values.device, dtype=torch.long)
    detached = values.detach()
    def multipliers(x):
        return box.to_material(x) / nominal_parameters(x.device)
    theta = multipliers(detached)
    seeds = torch.func.jacfwd(multipliers)(detached).index_select(1, indices)
    steps = model._steps(theta, duration)
    return _SeededAudio.apply(values, theta, seeds, indices, model, steps)
