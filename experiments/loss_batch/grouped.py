"""Experimental independent-item loss batching, preserving the summed loss.

This avoids retaining all loss graphs simultaneously. The simulator audio graph
is retained once; each loss chunk produces an upstream audio gradient and the
simulator receives their concatenation in one VJP. First derivative only.
"""
import torch


def grouped_loss_backward(audio, criterion, group):
    if audio.ndim != 2 or not isinstance(group, int) or group < 1:
        raise ValueError('Expected [batch, samples] audio and a positive group size.')
    bound = getattr(criterion, 'bound', None)
    if bound is None or not hasattr(bound, 'reference') or getattr(bound, 'reduction', None) != 'mean':
        raise ValueError('Expected a criterion bound to one target with mean reduction (CeL or waveform MAE)')
    if hasattr(bound, 'mass') and (bound.reference.ndim != 3 or bound.mass.numel() != 1):
        raise ValueError('Only a single common bound target is supported')
    if not hasattr(bound, 'mass') and bound.reference.shape[0] != 1:
        raise ValueError('Only a single common bound target is supported')
    gradients = []
    value = audio.new_zeros(())
    for start in range(0, audio.shape[0], group):
        chunk = audio[start:start+group].detach().requires_grad_(True)
        # Bound losses use reduction='mean': multiply by the number of items to
        # obtain the same sum as independent, scalar per-item losses.
        loss = criterion(chunk) * chunk.shape[0]
        grad, = torch.autograd.grad(loss, chunk)
        gradients.append(grad.detach())
        value = value + loss.detach()
    audio.backward(torch.cat(gradients, dim=0))
    return value
