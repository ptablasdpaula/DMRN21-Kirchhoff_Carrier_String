"""Experimental exact first-order default CeL VJP, one direction at a time.

Uses the installed bound STFT/CeL's saved target, window, mass, epsilon and
normalisation. It does not replace general CeL configurations or higher-order
autograd. Individual groups are reduced immediately to limit temporary memory.
"""
import math
import torch

_DIRECTIONS = ('right_up', 'right_down', 'left_up', 'left_down')


def _scan(x, dim, reverse):
    return x.flip((dim,)).cumsum(dim).flip((dim,)) if reverse else x.cumsum(dim)


def _surface(x, frequency_reverse, time_reverse):
    return _scan(_scan(x, -1, time_reverse), -2, frequency_reverse)


def _transpose(x, frequency_reverse, time_reverse):
    # Adjoint of a forward cumulative sum is a reverse cumulative sum.
    return _scan(_scan(x, -2, not frequency_reverse), -1, not time_reverse)


def manual_loss_backward(audio, criterion, group=16):
    """Accumulate sum of default per-item losses, then apply one audio VJP."""
    if audio.ndim != 2 or not isinstance(group, int) or group < 1 or not audio.requires_grad:
        raise ValueError('Expected differentiable [batch, samples] audio and group >= 1')
    bound = criterion.bound
    directions = tuple(getattr(x, 'value', x) for x in bound.directions)
    if bound.reference.ndim != 3 or bound.mass.numel() != 1:
        raise ValueError('Only a single common bound target is supported')
    if directions != _DIRECTIONS or bound.reduction != 'mean' or any(
            getattr(bound, field).numel() != 0 for field in
            ('frequency_forward','frequency_reverse','time_forward','time_reverse','sqrt_weights')):
        raise ValueError('Only the default unweighted diagonal CeL is supported')
    transform = criterion.transform
    if transform.center or transform.normalized:
        raise ValueError('Expected the default uncentered, unnormalized STFT')
    gradients = []
    total = audio.new_zeros(())
    for start in range(0, audio.shape[0], group):
        chunk = audio[start:start+group].detach().requires_grad_(True)
        power = transform(chunk)
        if not bool(torch.isfinite(power.detach()).all()):
            raise ValueError('Candidate energy must be finite')
        with torch.no_grad():
            grad_power = torch.zeros_like(power)
            root = math.sqrt(power.shape[-2] * power.shape[-1])
            mass = bound.mass
            for i, (fr, tr) in enumerate(((False,False),(True,False),(False,True),(True,True))):
                surface = _surface(power.detach(), fr, tr)
                normalized = surface / mass
                feature = normalized.clamp_min(bound.eps).sqrt()
                error = feature - bound.reference[i]
                norm = torch.linalg.vector_norm(error.flatten(start_dim=-2), dim=-1)
                total = total + (norm / (4 * root)).sum()
                safe_norm = norm.clamp_min(torch.finfo(norm.dtype).tiny)[:,None,None]
                d_surface = error / (safe_norm * (8 * root) * mass * feature)
                d_surface = torch.where((normalized >= bound.eps) & (norm[:,None,None] > 0), d_surface, 0.)
                grad_power.add_(_transpose(d_surface, fr, tr))
        grad, = torch.autograd.grad(power, chunk, grad_power)
        gradients.append(grad.detach())
    audio.backward(torch.cat(gradients, dim=0))
    return total
