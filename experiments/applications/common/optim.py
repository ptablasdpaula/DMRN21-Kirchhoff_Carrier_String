"""Batched Adam on bounded logits with per-item plateau decay and lowest-loss retention.

A port of ``fit_batch`` from ICASSP27-Phrase (``src/optimization.py``) to a
generic objective: every item of the batch has its own learning rate, its own
``ReduceLROnPlateau`` bookkeeping (reduce when the bad count exceeds the
patience, relative threshold, the reduction resets only the plateau counter),
an independent early-stopping counter, and the parameters at its strict lowest
loss are retained. Adam's update is invariant to the gradient scale, so the
per-item learning rate multiplies the step, not the gradient.
"""
from dataclasses import dataclass, field
import time
import torch

PLATEAU_EPSILON = 1e-8


@dataclass(frozen=True)
class Schedule:
    """Adam with plateau decay. The defaults assume ~1 s per evaluation and a few hundred evaluations."""
    initial_lr: float = 0.05
    factor: float = 0.5
    plateau_patience: int = 15
    threshold: float = 1e-3
    stop_patience: int = 60
    maximum_updates: int = 400
    betas: tuple = (0.9, 0.999)
    epsilon: float = 1e-8
    relative: bool = True   # divide each item's loss by its initial value before differentiating

    def __post_init__(self):
        if not (self.initial_lr > 0 and 0 < self.factor < 1 and 0 <= self.threshold < 1):
            raise ValueError('Invalid learning rate, factor or relative threshold.')
        if self.plateau_patience < 0 or self.stop_patience < 1 or self.maximum_updates < 1:
            raise ValueError('Invalid patience or update count.')
        if not all(0 <= beta < 1 for beta in self.betas) or self.epsilon <= 0:
            raise ValueError('Invalid Adam parameters.')


@dataclass
class FitResult:
    best_logits: torch.Tensor      # (B, n) logits at each item's strict lowest loss
    best_loss: torch.Tensor        # (B,)
    final_loss: torch.Tensor       # (B,) loss at the last evaluation of each item
    initial_loss: torch.Tensor     # (B,)
    updates: torch.Tensor          # (B,) Adam updates applied per item
    reductions: torch.Tensor       # (B,) learning-rate reductions per item
    events: list                   # per item, list of {update, old_lr, new_lr, best_loss}
    evaluations: int               # objective evaluations (iterations), including the initial one
    history: list = field(default_factory=list)
    seconds: float = 0.0


def fit_batch(logits, objective, *, schedule=Schedule(), callback=None, history_every=1):
    """Minimise ``objective`` independently for every row of ``logits``.

    ``objective(selected_logits, indices)`` receives the rows that are still
    active (``(K, n)`` with gradient) and their batch indices, and returns the
    per-item losses ``(K,)`` or a pair ``(losses, aux)``; ``aux`` (for example
    the rendered audio, detached) is passed to ``callback``. ``callback`` is
    called after every evaluation with a dict holding ``evaluation``,
    ``indices``, ``losses``, ``strict_best``, ``best_logits``, ``lr``,
    ``active``, ``updates`` and ``aux``.
    """
    if logits.ndim != 2 or logits.dtype != torch.float64:
        raise ValueError('logits must be a (batch, n) float64 tensor.')
    start = time.perf_counter()
    batch, device = logits.shape[0], logits.device
    raw = logits.detach().clone().requires_grad_(True)
    best_raw = raw.detach().clone()
    first, second = torch.zeros_like(raw), torch.zeros_like(raw)
    updates = torch.zeros(batch, dtype=torch.long, device=device)
    lr = torch.full((batch,), schedule.initial_lr, dtype=torch.float64, device=device)
    plateau_bad = torch.zeros(batch, dtype=torch.long, device=device)
    stop_bad = torch.zeros(batch, dtype=torch.long, device=device)
    reductions = torch.zeros(batch, dtype=torch.long, device=device)
    active = torch.ones(batch, dtype=torch.bool, device=device)
    initial_loss = scheduler_best = strict_best = final_loss = None
    events = [[] for _ in range(batch)]
    history = []
    beta1, beta2 = schedule.betas
    evaluations = 0
    while bool(active.any()):
        indices = active.nonzero().flatten()
        selected = raw[indices]
        out = objective(selected, indices)
        values, aux = (out if isinstance(out, tuple) else (out, None))
        if values.shape != (len(indices),):
            raise ValueError('objective must return one loss per active item.')
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('non-finite optimisation loss')
        detached = values.detach()
        evaluations += 1
        if initial_loss is None:
            initial_loss = detached.clone()
            scheduler_best = detached.clone()
            strict_best = detached.clone()
            final_loss = detached.clone()
            keep = torch.ones_like(values, dtype=torch.bool)
        else:
            final_loss[indices] = detached
            improved = detached < strict_best[indices]
            strict_best[indices[improved]] = detached[improved]
            best_raw[indices[improved]] = selected.detach()[improved]
            meaningful = detached < scheduler_best[indices]*(1-schedule.threshold)
            scheduler_best[indices] = torch.where(meaningful, detached, scheduler_best[indices])
            plateau_bad[indices] = torch.where(meaningful, torch.zeros_like(plateau_bad[indices]), plateau_bad[indices]+1)
            stop_bad[indices] = torch.where(meaningful, torch.zeros_like(stop_bad[indices]), stop_bad[indices]+1)
            # ReduceLROnPlateau reduces when num_bad_epochs > patience, not at equality.
            due = plateau_bad[indices] > schedule.plateau_patience
            if bool(due.any()):
                due_indices = indices[due]
                old = lr[due_indices].clone()
                proposed = old*schedule.factor
                changed = old-proposed > PLATEAU_EPSILON
                reduced = due_indices[changed]
                if len(reduced):
                    lr[reduced] = proposed[changed]
                    reductions[reduced] += 1
                    for previous, i in zip(old[changed].tolist(), reduced.tolist()):
                        events[i].append({'update': int(updates[i]), 'old_lr': previous,
                                          'new_lr': float(lr[i]), 'best_loss': float(strict_best[i])})
                # Only the plateau counter resets; the early-stopping counter continues.
                plateau_bad[due_indices] = 0
            stop = (stop_bad[indices] >= schedule.stop_patience) | (updates[indices] >= schedule.maximum_updates)
            keep = ~stop
            active[indices[stop]] = False
        if callback is not None:
            with torch.no_grad():
                callback({'evaluation': evaluations, 'indices': indices, 'losses': detached, 'strict_best': strict_best,
                          'best_logits': best_raw, 'lr': lr, 'active': active, 'updates': updates, 'aux': aux})
        if history_every and evaluations % history_every == 0:
            best_item = int(strict_best.argmin())
            history.append({'evaluation': evaluations, 'seconds': time.perf_counter()-start,
                            'loss_min': float(detached.min()), 'loss_median': float(detached.median()),
                            'best_min': float(strict_best.min()), 'best_item': best_item,
                            'lr_best_item': float(lr[best_item]), 'active': int(active.sum())})
        if not bool(keep.any()):
            continue
        step_indices = indices[keep]
        scale = initial_loss[step_indices] if schedule.relative else torch.ones_like(initial_loss[step_indices])
        (gradient_all,) = torch.autograd.grad((values[keep]/scale).sum(), selected)
        gradient = gradient_all[keep]
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('non-finite optimisation gradient')
        first[step_indices] = beta1*first[step_indices] + (1-beta1)*gradient
        second[step_indices] = beta2*second[step_indices] + (1-beta2)*gradient.square()
        updates[step_indices] += 1
        step = updates[step_indices].to(torch.float64)[:, None]
        delta = lr[step_indices, None]*(first[step_indices]/(1-beta1**step)) / (
            torch.sqrt(second[step_indices]/(1-beta2**step)) + schedule.epsilon)
        next_raw = raw.detach().clone()
        next_raw[step_indices] -= delta
        raw = next_raw.requires_grad_(True)
    return FitResult(best_raw, strict_best, final_loss, initial_loss, updates, reductions, events,
                     evaluations, history, time.perf_counter()-start)
