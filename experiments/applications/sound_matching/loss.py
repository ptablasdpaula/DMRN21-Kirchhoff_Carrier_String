"""Level-invariant Cumulative Energy Loss between rendered strings and a recorded note."""
import math
import torch
from cels import Direction, STFTCumulativeEnergyLoss

N_FFT, HOP = 1024, 256


def unit_rms(audio, eps=1e-12):
    """Remove the per-row mean and scale each row to unit RMS (differentiable)."""
    centred = audio - audio.mean(dim=-1, keepdim=True)
    return centred/(centred.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps)


VARIANTS = {
    'logcel': dict(log_weighting=True),                                                             # four diagonals, log-frequency weights
    'tlogcel': dict(log_weighting=True, directions=(Direction.RIGHT_UP, Direction.RIGHT_DOWN)),     # time-forward diagonals only
    'cel': dict(),                                                                                  # four diagonals, linear weights
}


def make_criterion(sample_rate=16000, variant='logcel'):
    """A frozen ICASSP27-Phrase CeL configuration, unreduced so every item keeps its own loss."""
    return STFTCumulativeEnergyLoss(sample_rate=sample_rate, n_fft=N_FFT, hop_length=HOP, reduction='none', **VARIANTS[variant])


def per_item_loss(criterion, candidate, target):
    """Per-row CeL between ``(B, S)`` candidates and a ``(S,)`` or ``(B, S)`` target, both at unit RMS."""
    candidate = unit_rms(candidate)
    target = unit_rms(torch.as_tensor(target, dtype=candidate.dtype, device=candidate.device))
    if target.ndim == 1:
        target = target[None].expand(candidate.shape[0], -1)
    return criterion(candidate, target)


def log_spectral_distance(a, b, n_fft=N_FFT, hop=HOP, floor_db=-80.0):
    """Mean over frames of the RMS over bins of the dB difference between two unit-RMS signals."""
    a, b = unit_rms(torch.as_tensor(a, dtype=torch.float64)), unit_rms(torch.as_tensor(b, dtype=torch.float64))
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    def db(x):
        spec = torch.stft(x, n_fft, hop, n_fft, window, center=True, return_complex=True).abs()
        return (20*torch.log10(spec.clamp_min(1e-12))).clamp_min(floor_db)
    return float(((db(a)-db(b))**2).mean(dim=0).sqrt().mean())
