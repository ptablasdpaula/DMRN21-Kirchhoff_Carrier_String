"""One full-band CeL loss, one fixed grid, one full-duration signal per fit."""
import math
import time
import json
from dataclasses import asdict
from pathlib import Path
import torch
from cels import STFTCumulativeEnergyLoss
from .parameters import NAMES

LOSS_COMMIT = '9fdbd60f5b67def8dd8884a006541e0ca4dbbf0d'


def criterion_for(target, sample_rate):
    if target.numel() < 1024:
        raise ValueError('Default CeL requires at least 1024 audio samples.')
    return STFTCumulativeEnergyLoss(sample_rate=sample_rate).bind(target.detach())


def should_stop(history):
    if len(history) <= 20:
        return False
    before = history[-21]['best_loss']
    return (before-history[-1]['best_loss'])/max(before,1e-30) < .1


def fit(renderer, target, initial, mask=(True,)*6, optimizer_name='adam',
        learning_rate=None, max_steps=500, progress=None, criterion=None):
    """First-order fit in bounded coordinates. No adaptive grid or feasibility loop.

    Endpoint values of selected parameters are nudged inside the box to obtain
    finite logits. Frozen parameters retain their exact supplied values.
    """
    box=renderer.grid.box
    box.validate(initial)
    if len(mask)!=6 or optimizer_name not in ('adam','lbfgs'):
        raise ValueError('Use a six-element selection mask and adam or lbfgs.')
    if not isinstance(max_steps,int) or max_steps < 0:
        raise ValueError('Maximum iterations must be a nonnegative integer.')
    learning_rate = (.01 if optimizer_name=='adam' else 1.) if learning_rate is None else learning_rate
    if not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Learning rate must be positive and finite.')
    if target.shape != (renderer.samples,) or not bool(torch.isfinite(target).all()):
        raise ValueError('Target must be finite and match the full render duration.')
    criterion=criterion_for(target,renderer.grid.sample_rate) if criterion is None else criterion
    initial=initial.detach().clone()
    indices=torch.tensor([i for i,selected in enumerate(mask) if selected],dtype=torch.long)
    base_logits=box.encode(initial)
    z=torch.nn.Parameter(base_logits[indices].clone())
    def parameters():
        decoded=box.decode(base_logits.scatter(0,indices,z))
        return torch.where(torch.tensor(mask,dtype=torch.bool),decoded,initial)
    optimizer=None
    if indices.numel():
        if optimizer_name=='adam': optimizer=torch.optim.Adam([z],lr=learning_rate)
        else: optimizer=torch.optim.LBFGS([z],lr=learning_rate,max_iter=1,max_eval=25,
                        history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-14)
    started=time.perf_counter();calls=0;history=[];best_loss=math.inf;best=initial.clone();cache={}
    effective_initial=parameters().detach()
    def evaluate():
        nonlocal calls,cache
        calls+=1
        if optimizer is not None: optimizer.zero_grad()
        p=parameters()
        value=criterion(renderer(p))
        if not bool(torch.isfinite(value)): raise RuntimeError('Nonfinite objective; fit stopped.')
        if optimizer is not None:
            value.backward()
            if not bool(torch.isfinite(z.grad).all()): raise RuntimeError('Nonfinite gradient; fit stopped.')
        cache=dict(z=z.detach().clone(),physical=p.detach().clone(),loss=value.item())
        return value
    def record(iteration):
        nonlocal best_loss,best
        if cache['loss']<best_loss:
            best_loss=cache['loss'];best=cache['physical'].clone()
        history.append(dict(iteration=iteration,loss=cache['loss'],best_loss=best_loss,
                            parameters=cache['physical'].tolist(),evaluations=calls,seconds=time.perf_counter()-started))
        if progress is not None: progress(history[-1])
    evaluate();record(0)
    reason='Reached maximum iteration count.'
    if optimizer is None: reason='No parameters selected.'
    elif best_loss==0: reason='Initial signal already has zero loss.'
    else:
        for iteration in range(1,max_steps+1):
            if optimizer_name=='adam': optimizer.step();evaluate()
            else:
                optimizer.step(evaluate)
                if not torch.equal(cache['z'],z.detach()): evaluate()
            record(iteration)
            if should_stop(history):
                reason='Less than 10% best-loss improvement over the last 20 optimizer steps.'
                break
    return dict(name=optimizer_name,best_physical=best,best_loss=best_loss,history=history,
                iterations=len(history)-1,evaluations=calls,seconds=time.perf_counter()-started,
                stop_reason=reason,mask=list(mask),requested_initial=initial,
                effective_initial=effective_initial,learning_rate=learning_rate)


def save_result(path, result, renderer, target_parameters):
    """Portable experiment evidence without embedding audio or executable objects."""
    payload={k:(v.tolist() if isinstance(v,torch.Tensor) else v) for k,v in result.items() if k!='audio'}
    payload.update(parameter_names=NAMES,target=target_parameters.tolist(),grid=asdict(renderer.grid),
                   duration=renderer.duration,excitation=renderer.excitation,
                   loss=dict(name='STFTCumulativeEnergyLoss',source_commit=LOSS_COMMIT,
                             sample_rate=renderer.grid.sample_rate,n_fft=1024,hop_length=256,log_weighting=False),
                   relative_error=((result['best_physical']-target_parameters)/target_parameters.clamp_min(1e-30)).tolist())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
